"""
Local LLM extraction  via an OpenAI-compatible endpoint

Works with either
    ollama serve -> http://localhost:11434/v1
    llama-server -m model.gguf --parallel 4 -> http://localhost/v1

One call per item. Batching items into one prompt makes the model want to 
fill every slot, which is exactly the failure we're trying to avoid. Locally
you pay in  seconds, not dollars, so buy the reliability.
"""

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

# Tight schemma. Every field costs output tokens, and vaguer fields invite 
# invention. `pain: false` means every other field must be null/empty.
SCHEMA = {
    "type": "object",
    "properties": {
        "pain":         {"type": "boolean"},
        "problem":      {"type": ["string", "null"]},
        "workaround":   {"type": ["string", "null"]},
        "tools":        {"type": "array", "items": {"type": "string"}},
        "role":         {"type": ["string", "null"]},
        "money":        {"type": "boolean"},
        "severity":     {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
    },
    "required": ["pain", "problem", "workaround", "tools", "role", "money", "severity"],
}

SYSTEM = """You extract operational pain points from Reddit posts by small-business owners.

MOST ITEMS CONTAIN NO PAIN POINT. Returning pain=false is the expected, normal
outcome.

Return pain=false for ALL of these, even when a real problem is described:
- PROMOTIONAL POSTS. If the author is announcing, launching, or describing a
  tool/service/system they built or use, it is promotion, not pain. A problem
  described as setup for a solution the author provides is pain=false.
- Research or interview write-ups ("what I learned talking to N people").
  The author is reporting others' problems, not their own.
- Hypothetical, past, or abandoned businesses ("I almost started", "if I ever").
- One-off incidents: a single dispute, a single bad customer, a single refund.
- Complaints about being wronged by a vendor (billing errors, bad support,
  a bad agency) rather than a recurring task in their own operations.
- Requests for recommendations, jokes, and general venting.

Return pain=true ONLY if ALL of these hold:
1. The author currently operates the business
2. The problem is a RECURRING task or workflow, not a one-time event
3. It costs them time, money, or customers on an ongoing basis

Fields when pain=true:
  problem    one plain sentence, normalized, no quotes from the text
  workaround what they CURRENTLY do, stated in the text. Advice they received,
             lessons learned, or plans are NOT workarounds. null if not stated.
  tools      software they use for this workflow. Not incidental mentions.
  role       ONLY if they state their current job or business type. Never infer
             from writing style, topic, or expertise.
  money      true ONLY if a specific cost, revenue, or budget figure or claim
             appears in the text
  severity   1 = minor annoyance, tolerable indefinitely
             2 = wastes under an hour a week
             3 = wastes several hours a week, or causes occasional lost revenue
             4 = a major recurring cost they have tried and failed to solve
             5 = threatens business viability. RARE. Almost never correct.

When pain=false: problem, workaround, role, severity are null and tools is [].

Reply with JSON only. /no_think"""


class LocalLLM:
    def __init__(self, base_url="http://localhost:11434/v1", model="qwen3:14b", cache_path="../data/llm_cache.jsonl", temperature=0.0, timeout=180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.cache_path = Path(cache_path)
        self._lock = threading.Lock()
        self._cache = self._load_cache()

    # Cache
    def _load_cache(self):
        cache = {}
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        record = json.loads(line)
                        cache[record["k"]] = record["v"]
        return cache

    def _key(self, text):
        blob = f"{self.model}|{SYSTEM}|{text}"
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def _remember(self, key, value):
        with self._lock:
            self._cache[key] = value
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as file:
                file.write(json.dumps({"k": key, "v": value}) + "\n")

    # Inference
    def _post(self, text):
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": text},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "pain", "schema": SCHEMA, "strict": True},
            },
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def extract_one(self, text, use_cache=True):
        """Returns (record_dict_or_None, status). Never raises."""
        key = self._key(text)
        if use_cache and key in self._cache:
            return self._cache[key], "cached"

        try:
            raw = self._post(text)
        except Exception as e:
            return None, f"http_error: {type(e).__name__}"

        record = parse_json(raw)
        if record is None:
            return None, "parse_error"

        record = enforce_nulls(record)
        self._remember(key, record)
        return record, "ok"

    def extract_many(self, texts, workers=4, on_progress=None):
        """Concurrent extraction. Match `workers` to llama-server --parallel."""
        results = [None] * len(texts)
        statuses = [None] * len(texts)
        done = [0]

        def work(i):
            record, status = self.extract_one(texts[i])
            results[i], statuses[i] = record, status
            done[0] += 1
            if on_progress and done[0] % 25 == 0:
                on_progress(done[0], len(texts))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, range(len(texts))))
        return results, statuses


# Parsing helpers
def parse_json(raw):
    """Tolerate code fences, prose preamble, and <think> blocks."""
    if not raw:
        return None
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def enforce_nulls(record):
    """
    A false `pain` must zero everything else.

    Small models often set pain=false then fill the fields anyway. The grammar
    can't catch this - it's valid JSON, just incoherent. So we enforce it.
    """
    record.setdefault("tools", [])
    if not record.get("pain"):
        return {
            "pain": False,
            "problem": None,
            "workaround": None,
            "tools": [],
            "role": None,
            "money": False,
            "severity": None,
        }

    for k in ("problem", "workaround", "role"):
        v = record.get(k)
        if isinstance(v, str) and v.strip().lower() in (
            "",
            "none",
            "null",
            "n/a",
            "not stated",
            "unknown",
        ):
            record[k] = None
    if not isinstance(record.get("tools"), list):
        record["tools"] = []
    return record


def health_check(base_url="http://localhost:11434/v1"):
    """Confirm the server is up and list what it's serving."""
    response = requests.get(f"{base_url.rstrip('/')}/models", timeout=10)
    response.raise_for_status()
    return [m["id"] for m in response.json().get("data", [])]