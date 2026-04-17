import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai

from data_loader import load_lettria, load_oskgc, sample_proportional

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env") # load env variables (API key, dataset paths)


def load_gemini_clients():
    """Load one or many Gemini API keys from env and return initialized clients."""
    keys = []

    # Preferred format: GEMINI_API_KEYS="key1,key2,key3"
    keys_csv = os.getenv("GEMINI_API_KEYS", "").strip()
    if keys_csv:
        keys.extend([k.strip() for k in keys_csv.split(",") if k.strip()])

    # Backward compatible: GEMINI_API_KEY="single_key"
    single_key = os.getenv("GEMINI_API_KEY", "").strip()
    if single_key and single_key not in keys:
        keys.append(single_key)

    if not keys:
        raise ValueError("No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEYS in .env")

    return [genai.Client(api_key=k) for k in keys]


clients = load_gemini_clients()

# Free tier constraints can be tuned from env if needed.
PER_KEY_RPM = max(1, int(os.getenv("GEMINI_PER_KEY_RPM", "5")))
PER_KEY_DAILY_LIMIT = max(1, int(os.getenv("GEMINI_PER_KEY_DAILY_LIMIT", "20")))
MIN_SECONDS_BETWEEN_CALLS = 60.0 / PER_KEY_RPM

_key_last_call_ts = [0.0] * len(clients)
_key_calls_today = [0] * len(clients)
_current_day = time.strftime("%Y-%m-%d")
_next_client_idx = 0


def _reset_daily_counters_if_needed():
    global _current_day, _key_calls_today
    today = time.strftime("%Y-%m-%d")
    if today != _current_day:
        _current_day = today
        _key_calls_today = [0] * len(clients)


def _reserve_next_client_slot():
    """Pick the next available client using round-robin + rate/daily limits."""
    global _next_client_idx

    _reset_daily_counters_if_needed()
    now = time.time()

    best_idx = None
    best_wait = None

    for offset in range(len(clients)):
        idx = (_next_client_idx + offset) % len(clients)

        if _key_calls_today[idx] >= PER_KEY_DAILY_LIMIT:
            continue

        wait = max(0.0, MIN_SECONDS_BETWEEN_CALLS - (now - _key_last_call_ts[idx]))
        if wait == 0.0:
            _next_client_idx = (idx + 1) % len(clients)
            return idx, 0.0

        if best_wait is None or wait < best_wait:
            best_idx = idx
            best_wait = wait

    if best_idx is not None:
        _next_client_idx = (best_idx + 1) % len(clients)
        return best_idx, best_wait

    return None, None


lettria_dir = PROJECT_ROOT / os.getenv("LETTRIA_DIR") # path to LettrIA dataset
oskgc_dir   = PROJECT_ROOT / os.getenv("OSKGC_DIR")   # path to OSKGC dataset


def has_connected_multihop(triples):
    """Return True if at least 2 triples are connected through a shared entity."""
    if len(triples) < 2:
        return False

    nodes_per_triple = []
    for t in triples:
        sub = str(t.get("sub", "")).strip()
        obj = str(t.get("obj", "")).strip()
        nodes_per_triple.append({sub, obj})

    for i in range(len(nodes_per_triple)):
        for j in range(i + 1, len(nodes_per_triple)):
            if nodes_per_triple[i] & nodes_per_triple[j]:
                return True
    return False


def default_multi_hop(reason="insufficient_connected_triples"):
    """Canonical placeholder when multi-hop is not possible from provided triples."""
    return {
        "question": None,
        "answer": None,
        "chain": None,
        "status": reason,
    }


def build_prompt(entry, allow_multihop):
    triples_str = "\n".join(
        f"- ({t['sub']}, {t['rel']}, {t['obj']})"
        for t in entry["triples"]
    )

    if allow_multihop:
        task_instructions = """Generate exactly 2 questions based STRICTLY on the triples above.

1. GRAPH_SINGLE_HOP: a question answerable using exactly ONE triple.
   - The answer must be the object or subject of a single triple.

2. GRAPH_MULTI_HOP: a question requiring traversal of AT LEAST 2 connected triples.
   - Two triples are "connected" if the object of one equals the subject of another.
   - Add a "chain" field showing each hop as: entity1 -[relation]-> entity2 -> ...
   - The answer must only be reachable by following this chain."""

        output_schema = """{
  "graph_single_hop": {"question": "...", "answer": "..."},
  "graph_multi_hop": {"question": "...", "answer": "...", "chain": "entity -[rel]-> entity -[rel]-> answer"}
}"""
    else:
        task_instructions = """Generate exactly 1 question based STRICTLY on the triples above.

1. GRAPH_SINGLE_HOP: a question answerable using exactly ONE triple.
   - The answer must be the object or subject of a single triple.

Do NOT generate a multi-hop question. The triples do not contain enough connected facts for multi-hop reasoning."""

        output_schema = """{
  "graph_single_hop": {"question": "...", "answer": "..."},
  "graph_multi_hop": null
}"""

    return f"""You are an expert in Knowledge Graph Question Answering.

Here is a knowledge graph entry with its associated triples.

Triples:
{triples_str}

{task_instructions}

Grounding rules (STRICT):
- Use ONLY entities and relations that explicitly appear in the triples above.
- Do NOT use world knowledge not present in the triples.
- Do NOT invent entities, relations, or answers.
- Every answer must be directly traceable to one or more triples.

Output ONLY valid JSON, no explanation, no markdown:
{output_schema}"""


def call_gemini(prompt, max_retries=3):
    """Send the prompt to Gemini and return the parsed JSON response"""
    for attempt in range(max_retries):
        client_idx, wait_time = _reserve_next_client_slot()
        if client_idx is None:
            print(f"  [✗] All API keys reached daily limit ({PER_KEY_DAILY_LIMIT}/day per key)")
            return None

        if wait_time and wait_time > 0:
            time.sleep(wait_time)

        # Count the reserved call now to stay conservative with free-tier quotas.
        _key_last_call_ts[client_idx] = time.time()
        _key_calls_today[client_idx] += 1

        try:
            response = clients[client_idx].models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            raw_text = response.text.strip()

            if raw_text.startswith("```"):
                # sometimes Gemini wraps the JSON in a markdown code block, strip it
                lines = raw_text.split("\n")
                raw_text = "\n".join(lines[1:-1])

            return json.loads(raw_text) # parse the JSON response

        except json.JSONDecodeError:
            print(f"  [!] Invalid JSON (attempt {attempt + 1}/{max_retries})")
            time.sleep(2) # wait before retrying
        except Exception as e:
            print(f"  [!] API error: {e} (attempt {attempt + 1}/{max_retries})")
            time.sleep(5) # longer wait on API error (rate limit, network, etc.)

    print("  [✗] Failed after 3 attempts")
    return None


def generate_questions(entries, dataset_name):
    """Run the LLM pipeline on dataset entries and return graph-only QA results."""
    results = []
    total = len(entries)

    for i, entry in enumerate(entries):
        print(f"  [{i+1}/{total}] {entry['id']} ({entry['category']})")

        allow_multihop = has_connected_multihop(entry["triples"])
        prompt = build_prompt(entry, allow_multihop) # build the prompt for this entry

        questions = call_gemini(prompt)
        if questions is None: # skip entry if all retries failed
            continue

        graph_multi_hop = questions.get("graph_multi_hop", {})
        if not allow_multihop:
            graph_multi_hop = default_multi_hop()

        results.append({
            "id":               f"{dataset_name}_{i+1:03d}", # unique id for the generated entry
            "source_id":        entry["id"],                  # original id from the dataset
            "dataset":          dataset_name,
            "category":         entry["category"],
            "original_sent":    entry["sent"],                # original sentence used to generate questions
            "original_triples": entry["triples"],             # knowledge graph triples used to generate questions
            "graph_single_hop": questions.get("graph_single_hop", {}),
            "graph_multi_hop":  graph_multi_hop,
        })

    return results


def save_results(results, output_path):
    """Save results as a .jsonl file (one JSON object per line)"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # create output directory if it doesn't exist
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n") # write each result as a single JSON line
    print(f"  → Saved: {output_path} ({len(results)} entries)")



OUTPUT_DIR = "output_questions"
SAMPLES_PER_DATASET = 100

print("DATASET: LettrIA")
lettria_sample = sample_proportional(load_lettria(lettria_dir), SAMPLES_PER_DATASET)
print("Generating questions...")
lettria_results = generate_questions(lettria_sample, "lettria")
save_results(lettria_results, f"{OUTPUT_DIR}/questions_graph_lettria.jsonl")

print("DATASET: OSKGC")
oskgc_sample = sample_proportional(load_oskgc(oskgc_dir), SAMPLES_PER_DATASET)
print("Generating questions...")
oskgc_results = generate_questions(oskgc_sample, "oskgc")
save_results(oskgc_results, f"{OUTPUT_DIR}/questions_graph_oskgc.jsonl")

print("DONE")
print(f"Total: {len(lettria_results) + len(oskgc_results)} graph entries processed")
