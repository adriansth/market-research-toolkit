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
import os
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
        "experience":   {"type": ["string", "null"], "enum": ["new", "experienced", "unknown", None]},
        "money":        {"type": "boolean"},
        "severity":     {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
    },
    "required": ["pain", "problem", "workaround", "tools", "role", "money", "severity"],
}

SYSTEM = """You extract collector pain points from Reddit posts by Pokemon card collectors and buyers.

MOST ITEMS CONTAIN NO PAIN POINT. Returning pain=false is the expected, normal
outcome.

Return pain=false for ALL of these, even when frustration is expressed:
- Collection showcases, pull posts, "look what I got" - however excited.
- Bare price checks with no reasoning ("what's this worth?"). But if the author
  explains WHY they cannot figure it out, that reasoning is pain=true.
- Complaints about scalpers, print runs, or The Pokemon Company. Real, but not
  something a tool could fix.
- Promotional posts. Anyone announcing, launching, or steering readers toward
  an app, shop, Discord, or service - including a problem described as setup
  for their solution.
- Advice-giving replies. Explaining what collectors in general should do is
  commentary, not the author's own pain.
- Nostalgia, deck discussion, gameplay, and general hobby chat.

Return pain=true ONLY if:
1. The author is personally collecting or buying
2. They describe a specific difficulty in acquiring, valuing, verifying,
   tracking, storing, grading, or selling cards
3. It cost them money, time, or a decision they regret - or is actively
   blocking one

Fields when pain=true:
  problem     one plain sentence, normalized, no quotes from the text
  workaround  what they CURRENTLY do about it, stated in the text. Advice they
              received or plans are NOT workarounds. null if not stated.
  tools       named SOFTWARE, apps, sites, or services only. Never cards, sets,
              sleeves, binders, or any physical product.
  experience  "new" if they say they are new, returning, or just started
              "experienced" if they describe years of collecting or expertise
              "unknown" if not stated. Never infer from writing style.
  money       true ONLY if a specific amount, price, or loss appears
  severity    1 = mild annoyance
              2 = wasted an hour or a few dollars
              3 = a recurring hassle, or a meaningful sum lost once
              4 = a large loss, or something that made them stop or scale back
              5 = quit the hobby over it. RARE.

When pain=false: problem, workaround, experience, severity are null, tools is [].

Reply with JSON only. /no_think"""


class LocalLLM:
    def __init__(self, base_url=None, model="qwen3:14b", cache_path="../data/llm_cache.jsonl", temperature=0.0, timeout=180):
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434/v1")).rstrip("/")
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
        """
        Concurrent extraction.

        on_progress(done, total, status, worker_name) fires after every item,
        under a lock so counts and output stay ordered.
        """
        results = [None] * len(texts)
        statuses = [None] * len(texts)
        counter = {"done": 0}
        lock = threading.Lock()

        def work(i):
            record, status = self.extract_one(texts[i])
            results[i], statuses[i] = record, status
            with lock:
                counter["done"] += 1
                if on_progress:
                    on_progress(counter["done"], len(texts), status, threading.current_thread().name)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="w") as pool:
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