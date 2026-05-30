import json
import csv
import re
from pathlib import Path


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# Abstention / "coverage error" patterns. The LLM output is ~99% English, plus a
# few French fallbacks. When the model declines to answer (because the retrieved
# context is insufficient), FactCC still scores the sentence — but an abstention is
# neither faithful nor hallucinated. Counting it in the faithfulness mean biases the
# metric, so we flag these rows and exclude them, reporting them separately as a
# "coverage error rate".
_COVERAGE_ERROR_PATTERNS = [
    # English
    r"i don.?t know",
    r"i do not know",
    r"i.?m not sure",
    r"i am not sure",
    r"i cannot determine",
    r"i can.?t determine",
    r"(the |provided |given )?context (does not|doesn.?t) (provide|contain|mention|include|specify|allow)",
    r"(the |provided |given )?context (is )?(insufficient|not (enough|sufficient))",
    r"based on the (provided |given )?context.{0,40}(cannot|can.?t|no|unable|not (enough|able|provided))",
    r"(there is |there.?s )?no (relevant )?information",
    r"no (relevant )?(information|details?|context|data|mention)",
    r"not (enough|sufficient) (information|context|details?)",
    r"insufficient (information|context|details?)",
    r"(cannot|can.?t|unable to) (answer|determine|find|provide|be (determined|answered))",
    r"(is |are )?not (mentioned|specified|provided|stated|available|present|found) (in|within) the (provided |given )?(context|text|passage|document)",
    r"(does not|doesn.?t|do not|don.?t) (mention|specify|state|provide|contain)",
    r"not (mentioned|specified|stated|provided|available|found)\b",
    r"no answer (can|could) be",
    # French fallbacks
    r"je ne sais pas",
    r"le contexte ne (permet|fournit|contient|mentionne|précise)",
    r"aucune information",
    r"pas (d.?information|assez d.?information)",
    r"(n.?est|ne sont) pas (mentionn|précis|indiqu|fourni)",
    r"impossible de (répondre|déterminer)",
]
_COVERAGE_ERROR_RE = re.compile("|".join(_COVERAGE_ERROR_PATTERNS), re.IGNORECASE)


def is_coverage_error(text) -> bool:
    """True if the answer is an abstention (model declined for lack of context).

    Such rows are a coverage failure, not a faithfulness measurement, and must be
    excluded from the faithfulness mean and reported separately.
    """
    if not text:
        return True  # empty answer == nothing was covered
    # Normalise curly/typographic apostrophes so `.?` apostrophe classes match.
    s = str(text).strip().replace("’", "'").replace("ʼ", "'")
    if not s:
        return True
    return bool(_COVERAGE_ERROR_RE.search(s))


def extract_rag_source(context):
    """
    Context RAG is a list of dicts with a 'text' field.
    We extract only the text, without metadata (id, distance...).
    """
    if isinstance(context, list):
        texts = []
        for chunk in context:
            if isinstance(chunk, dict):
                texts.append(chunk.get("text", ""))
            else:
                texts.append(str(chunk))
        return " ".join(t for t in texts if t)
    if isinstance(context, dict):
        return context.get("text", "")
    return str(context)


def extract_graphrag_source(context) -> str:
    """
    The GraphRAG context is a list of triples (subject, relation, object)
    (e.g., 'Arem arem -[origin]-> Indonesia').
    We deduplicate to remove repeated triples from FAISS.
    """
    if isinstance(context, list):
        seen = []
        for triple in context:
            t = str(triple).strip()
            if t and t not in seen:
                seen.append(t)
        return " ".join(seen)
    return str(context)


def build_source_index():
    """
    Build two lookup indexes from Task 1 question files:
      - t5_index:  id → original sentence  (for hop_type="t5")
      - llm_index: id → original sentence  (for hop_type="single_hop" / "multi_hop")

    Two separate indexes are needed because for OSKGC, the same id exists in both the
    T5 and LLM question files but refers to different source documents.
    """
    base = Path("task1_questions_generation/output")

    t5_index = {}
    for fname in ("questions_t5_lettria.jsonl", "questions_t5_oskgc.jsonl"):
        for e in load_jsonl(base / fname):
            t5_index[e["id"]] = e.get("sentence", "")

    llm_index = {}
    for fname in ("questions_llm_lettria.jsonl", "questions_llm_oskgc.jsonl"):
        for e in load_jsonl(base / fname):
            llm_index[e["id"]] = e.get("original_sent", "")

    return t5_index, llm_index


def get_original_source(entry, t5_index, llm_index):
    """Return the original source sentence for a pipeline entry."""
    hop = entry.get("hop_type", "t5")
    eid = entry["id"]
    if hop == "t5":
        source = t5_index.get(eid, "")
    else:  # single_hop or multi_hop
        source = llm_index.get(eid, "")
    if not source:
        print(f"  [WARNING] No source found for id={eid}, hop_type={hop}")
    return source


def convert_to_mirage(results_path, output_path, pipeline_name, t5_index, llm_index):
    '''Convert result JSONL from task2_setup_rag into CSV for MIRAGE (FactCC)'''
    entries = load_jsonl(results_path)
    rows = []

    for entry in entries:
        source = get_original_source(entry, t5_index, llm_index)

        rows.append({
            "source":           source,
            "gen":              entry["llm_answer"],
            "id":               entry["id"],
            "question":         entry["question"],
            "ground_truth":     entry.get("answer"),
            "hop_type":         entry.get("hop_type", "unknown"),
            "pipeline":         pipeline_name,
            "is_coverage_error": is_coverage_error(entry["llm_answer"]),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    n_cov = sum(1 for r in rows if r["is_coverage_error"])
    print(f"[{pipeline_name}] {len(rows)} entries → {output_path}  "
          f"({n_cov} coverage errors flagged, {n_cov / max(len(rows), 1):.1%})")


if __name__ == "__main__":
    t5_index, llm_index = build_source_index()

    convert_to_mirage(
        results_path="task2_setup_rag/output/rag_results.jsonl",
        output_path="task3_evaluation/inputs/rag_input.csv",
        pipeline_name="rag",
        t5_index=t5_index,
        llm_index=llm_index,
    )
    convert_to_mirage(
        results_path="task2_setup_rag/output/graphrag_results.jsonl",
        output_path="task3_evaluation/inputs/graphrag_input.csv",
        pipeline_name="graphrag",
        t5_index=t5_index,
        llm_index=llm_index,
    )
    convert_to_mirage(
        results_path="task2_setup_rag/output/rag_hybrid_results.jsonl",
        output_path="task3_evaluation/inputs/rag_hybrid_input.csv",
        pipeline_name="rag_hybrid",
        t5_index=t5_index,
        llm_index=llm_index,
    )