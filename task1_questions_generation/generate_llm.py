import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from data_loader import load_lettria, load_oskgc, sample_proportional

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env") # load env variables (API key, dataset paths)

api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key:
    raise ValueError("No OpenAI API key found. Set OPENAI_API_KEY in .env")

client = OpenAI(api_key=api_key)

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


SYSTEM_MSG = (
    "You are an expert in Knowledge Graph Question Answering.\n"
    "Your task is to generate grounded questions strictly from provided knowledge graph triples.\n"
    "Never use external knowledge."
)

USER_MSG_TEMPLATE = """Here is a knowledge graph entry.

Example:
Triples:
- (Paris, capital_of, France)
- (France, continent, Europe)

Output:
{{
  "graph_single_hop": {{"question": "What is Paris the capital of?", "answer": "France"}},
  "graph_multi_hop": {{"question": "On which continent is the country whose capital is Paris?", "answer": "Europe", "chain": "Paris -[capital_of]-> France -[continent]-> Europe"}}
}}

Now process this entry:

Triples:
{triples}

Generate exactly 2 questions based STRICTLY on the triples above. No external knowledge.

1. GRAPH_SINGLE_HOP: simple question, answer from exactly 1 triple.
2. GRAPH_MULTI_HOP: complex question, answer requires traversing 2+ connected triples.
   Chain format: EntityA -[relation1]-> EntityB -[relation2]-> EntityC

Grounding rules:
- Use ONLY entities and relations that explicitly appear in the triples above.
- Never use world knowledge not present in the triples.
- Every answer must be directly traceable to one or more triples.

Output ONLY valid JSON, no explanation, no markdown:
{{
  "graph_single_hop": {{"question": "...", "answer": "..."}},
  "graph_multi_hop": {{"question": "...", "answer": "...", "chain": "..."}}
}}"""

USER_MSG_SINGLE_ONLY_TEMPLATE = """Here is a knowledge graph entry.

Triples:
{triples}

Generate exactly 1 question based STRICTLY on the triples above. No external knowledge.

1. GRAPH_SINGLE_HOP: simple question, answer from exactly 1 triple.

Grounding rules:
- Use ONLY entities and relations that explicitly appear in the triples above.
- Never use world knowledge not present in the triples.
- Every answer must be directly traceable to one or more triples.

Output ONLY valid JSON, no explanation, no markdown:
{{
  "graph_single_hop": {{"question": "...", "answer": "..."}},
  "graph_multi_hop": null
}}"""


def build_prompt(entry, allow_multihop):
    triples_str = "\n".join(
        f"- ({t['sub']}, {t['rel']}, {t['obj']})"
        for t in entry["triples"]
    )
    template = USER_MSG_TEMPLATE if allow_multihop else USER_MSG_SINGLE_ONLY_TEMPLATE
    return SYSTEM_MSG, template.format(triples=triples_str)


def call_gpt4(system_msg, user_msg, max_retries=5):
    """Send the prompt to GPT-4 and return the parsed JSON response."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
            )
            raw_text = response.choices[0].message.content.strip()

            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                raw_text = "\n".join(lines[1:-1])

            return json.loads(raw_text)

        except json.JSONDecodeError:
            print(f"  [!] Invalid JSON (attempt {attempt + 1}/{max_retries})")
            time.sleep(2)
        except Exception as e:
            backoff = min(10 * (2 ** attempt), 60)
            print(f"  [!] GPT-4 error (attempt {attempt + 1}/{max_retries}), retrying in {backoff}s: {e}")
            time.sleep(backoff)

    print("  [✗] Failed after all attempts")
    return None


def generate_questions(entries, dataset_name):
    """Run the LLM pipeline on dataset entries and return graph-only QA results."""
    results = []
    total = len(entries)

    for i, entry in enumerate(entries):
        print(f"  [{i+1}/{total}] {entry['id']} ({entry['category']})")

        allow_multihop = has_connected_multihop(entry["triples"])
        system_msg, user_msg = build_prompt(entry, allow_multihop)

        questions = call_gpt4(system_msg, user_msg)
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



OUTPUT_DIR = Path(__file__).parent / "output_questions"
SAMPLES_PER_DATASET = 50

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
