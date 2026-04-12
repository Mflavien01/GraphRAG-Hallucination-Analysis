"""Question generation using Gemini LLM.

Generates 4 structured question types per entry (text/graph × single/multi-hop)
based on the sentence and its knowledge graph triples.
"""
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai

from data_loader import load_lettria, load_oskgc, sample_proportional

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env") # load env variables (API key, dataset paths)

client      = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) # initialize Gemini client with API key
lettria_dir = PROJECT_ROOT / os.getenv("LETTRIA_DIR") # path to LettrIA dataset
oskgc_dir   = PROJECT_ROOT / os.getenv("OSKGC_DIR")   # path to OSKGC dataset


def build_prompt(entry):
    """Build the prompt sent to Gemini for a given dataset entry"""
    triples_str = "\n".join(
        f"- ({t['sub']}, {t['rel']}, {t['obj']})"
        for t in entry["triples"] # format each triple as (subject, relation, object)
    )

    # ask Gemini to generate 4 types of questions based strictly on the sentence and triples
    # 2 text-based (single-hop and multi-hop) + 2 graph-based (single-hop and multi-hop)
    return f"""You are an expert in Knowledge Graph Question Answering.

Here is a real knowledge graph entry.

Sentence: "{entry['sent']}"

Triples:
{triples_str}

Generate exactly 4 questions based STRICTLY on the data above. No external knowledge.

1. TEXT_SINGLE_HOP: simple question, answer from the sentence, 1 fact only.
2. TEXT_MULTI_HOP: complex question, answer requires combining 2+ facts from the sentence.
3. GRAPH_SINGLE_HOP: simple question, answer from exactly 1 triple.
4. GRAPH_MULTI_HOP: complex question, answer requires traversing 2+ connected triples.

For multi-hop questions, add a "chain" field showing reasoning steps using → between each hop.

Output ONLY valid JSON, no explanation, no markdown:
{{
  "text_single_hop": {{"question": "...", "answer": "..."}},
  "text_multi_hop": {{"question": "...", "answer": "...", "chain": "..."}},
  "graph_single_hop": {{"question": "...", "answer": "..."}},
  "graph_multi_hop": {{"question": "...", "answer": "...", "chain": "..."}}
}}"""


def call_gemini(prompt, max_retries=3):
    """Send the prompt to Gemini and return the parsed JSON response"""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
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
    """Run the full LLM pipeline on a list of dataset entries and return structured results"""
    results = []
    total = len(entries)

    for i, entry in enumerate(entries):
        print(f"  [{i+1}/{total}] {entry['id']} ({entry['category']})")

        prompt = build_prompt(entry) # build the prompt for this entry

        # respect free-tier rate limit (10 req/min) — wait 7s between each call
        if i > 0:
            time.sleep(7)

        questions = call_gemini(prompt)
        if questions is None: # skip entry if all retries failed
            continue

        results.append({
            "id":               f"{dataset_name}_{i+1:03d}", # unique id for the generated entry
            "source_id":        entry["id"],                  # original id from the dataset
            "dataset":          dataset_name,
            "category":         entry["category"],
            "original_sent":    entry["sent"],                # original sentence used to generate questions
            "original_triples": entry["triples"],             # knowledge graph triples used to generate questions
            "text_single_hop":  questions.get("text_single_hop", {}),
            "text_multi_hop":   questions.get("text_multi_hop", {}),
            "graph_single_hop": questions.get("graph_single_hop", {}),
            "graph_multi_hop":  questions.get("graph_multi_hop", {}),
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

print("DATASET: LettrIA")
lettria_sample = sample_proportional(load_lettria(lettria_dir), 50) # load and sample 50 entries proportionally across categories
print("Generating questions...")
lettria_results = generate_questions(lettria_sample, "lettria")
save_results(lettria_results, f"{OUTPUT_DIR}/questions_lettria.jsonl")

print("DATASET: OSKGC")
oskgc_sample = sample_proportional(load_oskgc(oskgc_dir), 50) # same for OSKGC
print("Generating questions...")
oskgc_results = generate_questions(oskgc_sample, "oskgc")
save_results(oskgc_results, f"{OUTPUT_DIR}/questions_oskgc.jsonl")

print("DONE")
print(f"Total: {len(lettria_results) + len(oskgc_results)} questions generated")
