from pathlib import Path
import argparse
import json
import os
import sys
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
QUESTIONS_DIR = PROJECT_ROOT / "task1_questions_generation" / "output"
OUTPUT_DIR    = Path(__file__).parent / "output"

# rag/, graph_rag/, llm/ are loaded as namespace packages
sys.path.insert(0, str(Path(__file__).parent))


# ── JSON encoder to handle numpy types (e.g. float32 in RAG context/distances) ─
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ── question loaders ────────────────────────────────────────────────────────────
def load_questions_t5(file_path):
    """Load T5-generated QA pairs. Each entry has multiple qa_pairs per sentence."""
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            for qa in entry["qa_pairs"]:
                if qa["question"] and qa["answer"]:
                    questions.append({
                        "id":        entry["id"],
                        "source_id": entry["source_id"],
                        "dataset":   entry["dataset"],
                        "category":  entry["category"],
                        "hop_type":  "t5",
                        "question":  qa["question"],
                        "answer":    qa["answer"],
                    })
    return questions


def load_questions_llm(file_path):
    """Load LLM-generated graph questions (single-hop and multi-hop)."""
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)

            single_hop = entry.get("graph_single_hop") or {}
            if single_hop.get("question") and single_hop.get("answer"):
                questions.append({
                    "id":        entry["id"],
                    "source_id": entry["source_id"],
                    "dataset":   entry["dataset"],
                    "category":  entry["category"],
                    "hop_type":  "single_hop",
                    "question":  single_hop["question"],
                    "answer":    single_hop["answer"],
                })

            multi_hop = entry.get("graph_multi_hop") or {}
            if multi_hop.get("question") and multi_hop.get("answer"):
                questions.append({
                    "id":        entry["id"],
                    "source_id": entry["source_id"],
                    "dataset":   entry["dataset"],
                    "category":  entry["category"],
                    "hop_type":  "multi_hop",
                    "question":  multi_hop["question"],
                    "answer":    multi_hop["answer"],
                    "chain":     multi_hop.get("chain"),
                })
    return questions


# ── pipeline runner ─────────────────────────────────────────────────────────────
def run_and_save(questions, pipeline_fn, llm, output_file, pipeline_name):
    """Run pipeline_fn on each question and save results to output_file as JSONL."""
    results = []
    for i, q in enumerate(questions):
        print(f"[{pipeline_name}] [{i+1}/{len(questions)}] {q['id']} ({q['hop_type']})")
        output = pipeline_fn(q["question"], llm)
        results.append({
            "id":        q["id"],
            "source_id": q["source_id"],
            "dataset":   q["dataset"],
            "category":  q["category"],
            "hop_type":  q["hop_type"],
            "question":  q["question"],
            "answer":    q["answer"],
            **output
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, cls=_NumpyEncoder) + "\n")
    print(f"Saved {len(results)} results to {output_file}")


# ── individual pipelines ────────────────────────────────────────────────────────
# Each one is wrapped in a function so the heavy imports (FAISS index build, model
# download) only run for the pipeline you actually launched.

def run_rag_pipeline(llm, limit=None):
    """RAG on T5 questions only."""
    from rag.pipeline import run_rag

    questions  = load_questions_t5(QUESTIONS_DIR / "questions_t5_lettria.jsonl")
    questions += load_questions_t5(QUESTIONS_DIR / "questions_t5_oskgc.jsonl")
    print(f"Loaded {len(questions)} T5 questions")

    if limit:
        questions = questions[:limit]
        print(f"  → truncated to {len(questions)} for this run")

    if not questions:
        print("No T5 questions found — skipping RAG.")
        return

    run_and_save(questions, run_rag, llm, OUTPUT_DIR / "rag_results.jsonl", "RAG")


def run_graphrag_pipeline(llm, limit=None):
    """GraphRAG on LLM (single-hop + multi-hop) questions only."""
    from graph_rag.pipeline import run_graphrag

    questions = []
    for fname in ["questions_llm_lettria.jsonl", "questions_llm_oskgc.jsonl"]:
        fpath = QUESTIONS_DIR / fname
        if fpath.exists():
            questions += load_questions_llm(fpath)
    print(f"Loaded {len(questions)} LLM graph questions")

    if limit:
        questions = questions[:limit]
        print(f"  → truncated to {len(questions)} for this run")

    if not questions:
        print("No LLM questions found — skipping GraphRAG.")
        return

    run_and_save(questions, run_graphrag, llm, OUTPUT_DIR / "graphrag_results.jsonl", "GraphRAG")


# ── main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG and/or GraphRAG over the task 1 questions.")
    parser.add_argument("pipeline", nargs="?", choices=["rag", "graphrag", "all"], default="all",
                        help="which pipeline to run (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of questions per pipeline (useful for smoke tests)")
    args = parser.parse_args()

    from llm.llm_interface import QwenLLM
    llm = QwenLLM(token=os.getenv("HF_TOKEN"))

    if args.pipeline in ("rag", "all"):
        run_rag_pipeline(llm, limit=args.limit)

    if args.pipeline in ("graphrag", "all"):
        run_graphrag_pipeline(llm, limit=args.limit)
