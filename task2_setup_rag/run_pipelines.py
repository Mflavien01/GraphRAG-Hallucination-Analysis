from pathlib import Path
from dotenv import load_dotenv
import argparse
import json
import os
import sys
import time
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
QUESTIONS_DIR = PROJECT_ROOT / "task1_questions_generation" / "output"
OUTPUT_DIR    = Path(__file__).parent / "output"

load_dotenv(PROJECT_ROOT / ".env")

# rag/, graph_rag/, llm/ are loaded as namespace packages
sys.path.insert(0, str(Path(__file__).parent))


# ── retry helper ───────────────────────────────────────────────────────────────
# When the Eurecom gateway is offline, calls fail with 404 ("model does not exist")
# or with connection errors. Instead of crashing, we wait and retry — so you can
# leave the script running and it will pick up automatically when the GPU is back.
# Auth errors (401/403) are NOT retried since waiting won't fix them.
def _call_with_retry(fn, *args, **kwargs):
    delay = 30          # start small
    delay_max = 300     # cap at 5 minutes
    while True:
        try:
            return fn(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in (401, 403):
                print(f"\n  ✗ Authentication error ({status}) — fix EURECOM_LLM_KEY in .env, aborting.")
                raise
            print(f"\n  ⚠ LLM call failed: {type(e).__name__}: {str(e)[:200]}")
            print(f"  → gateway likely offline. Sleeping {delay}s then retrying… (Ctrl-C to abort, progress is saved)")
            try:
                time.sleep(delay)
            except KeyboardInterrupt:
                raise
            delay = min(delay * 2, delay_max)


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
    """Run pipeline_fn on each question and append results to output_file as JSONL.

    Each line is flushed right after the LLM call so a crash mid-run doesn't lose
    completed work. On restart, questions already present in the file (matched on
    id + hop_type) are skipped. LLM calls are wrapped in retry-with-backoff so the
    script survives the gateway being offline.
    Delete the file to start fresh.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    done = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["id"], r["hop_type"]))
        if done:
            print(f"  → resuming: {len(done)} results already in {output_file.name}")

    written = 0
    with open(output_file, "a", encoding="utf-8") as f:
        for i, q in enumerate(questions):
            if (q["id"], q["hop_type"]) in done:
                continue
            print(f"[{pipeline_name}] [{i+1}/{len(questions)}] {q['id']} ({q['hop_type']})")
            output = _call_with_retry(pipeline_fn, q["question"], llm)
            record = {
                "id":        q["id"],
                "source_id": q["source_id"],
                "dataset":   q["dataset"],
                "category":  q["category"],
                "hop_type":  q["hop_type"],
                "question":  q["question"],
                "answer":    q["answer"],
                **output
            }
            f.write(json.dumps(record, ensure_ascii=False, cls=_NumpyEncoder) + "\n")
            f.flush()
            written += 1

    print(f"Wrote {written} new results to {output_file} (total: {len(done) + written})")


# ── question loading ────────────────────────────────────────────────────────────
# Both pipelines run on the *same* question set (T5 + LLM, lettria + oskgc) so that
# MIRAGE in task 3 can compare RAG vs GraphRAG answers on identical inputs.
def load_all_questions():
    questions = []
    for fname in ["questions_t5_lettria.jsonl", "questions_t5_oskgc.jsonl"]:
        fpath = QUESTIONS_DIR / fname
        if fpath.exists():
            questions += load_questions_t5(fpath)
    for fname in ["questions_llm_lettria.jsonl", "questions_llm_oskgc.jsonl"]:
        fpath = QUESTIONS_DIR / fname
        if fpath.exists():
            questions += load_questions_llm(fpath)
    return questions


# ── individual pipelines ────────────────────────────────────────────────────────
# Each one is wrapped in a function so the heavy imports (FAISS index build, model
# download) only run for the pipeline you actually launched.

def run_rag_pipeline(llm, limit=None):
    from rag.pipeline import run_rag

    questions = load_all_questions()
    print(f"Loaded {len(questions)} questions (T5 + LLM, both datasets)")

    if limit:
        questions = questions[:limit]
        print(f"  → truncated to {len(questions)} for this run")

    if not questions:
        print("No questions found — skipping RAG.")
        return

    run_and_save(questions, run_rag, llm, OUTPUT_DIR / "rag_results.jsonl", "RAG")


def run_graphrag_pipeline(llm, limit=None):
    from graph_rag.pipeline import run_graphrag

    questions = load_all_questions()
    print(f"Loaded {len(questions)} questions (T5 + LLM, both datasets)")

    if limit:
        questions = questions[:limit]
        print(f"  → truncated to {len(questions)} for this run")

    if not questions:
        print("No questions found — skipping GraphRAG.")
        return

    run_and_save(questions, run_graphrag, llm, OUTPUT_DIR / "graphrag_results.jsonl", "GraphRAG")


def run_rag_hybrid_pipeline(llm, limit=None):
    from rag.pipeline_hybrid import run_rag_hybrid

    questions = load_all_questions()
    print(f"Loaded {len(questions)} questions (T5 + LLM, both datasets)")

    if limit:
        questions = questions[:limit]
        print(f"  → truncated to {len(questions)} for this run")

    if not questions:
        print("No questions found — skipping RAG Hybrid.")
        return

    run_and_save(
        questions,
        run_rag_hybrid,
        llm,
        OUTPUT_DIR / "rag_hybrid_results.jsonl",
        "RAG-Hybrid"
    )


def run_graphrag_hybrid_pipeline(llm, limit=None):
    from graph_rag.pipeline_hybrid import run_graphrag_hybrid

    questions = load_all_questions()
    print(f"Loaded {len(questions)} questions (T5 + LLM, both datasets)")

    if limit:
        questions = questions[:limit]
        print(f"  → truncated to {len(questions)} for this run")

    if not questions:
        print("No questions found — skipping GraphRAG Hybrid.")
        return

    run_and_save(
        questions,
        run_graphrag_hybrid,
        llm,
        OUTPUT_DIR / "graphrag_hybrid_results.jsonl",
        "GraphRAG-Hybrid"
    )


def run_graph_chunk_pipeline(llm, limit=None):
    from graph_rag.pipeline_graph_chunk import run_graph_chunk

    questions = load_all_questions()
    print(f"Loaded {len(questions)} questions (T5 + LLM, both datasets)")

    if limit:
        questions = questions[:limit]
        print(f"  → truncated to {len(questions)} for this run")

    if not questions:
        print("No questions found — skipping Graph-Chunk Hybrid.")
        return

    run_and_save(
        questions,
        run_graph_chunk,
        llm,
        OUTPUT_DIR / "graph_chunk_results.jsonl",
        "Graph-Chunk"
    )


def run_parent_child_pipeline(llm, limit=None):
    from rag.pipeline_parent_child import run_rag_parent_child

    questions = load_all_questions()
    print(f"Loaded {len(questions)} questions (T5 + LLM, both datasets)")

    if limit:
        questions = questions[:limit]
        print(f"  → truncated to {len(questions)} for this run")

    if not questions:
        print("No questions found — skipping RAG Parent-Child.")
        return

    run_and_save(
        questions,
        run_rag_parent_child,
        llm,
        OUTPUT_DIR / "rag_parent_child_results.jsonl",
        "RAG-Parent-Child"
    )

# ── main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG and/or GraphRAG over the task 1 questions.")
    parser.add_argument("pipeline", nargs="?",
                        choices=["rag", "graphrag", "hybrid", "graphrag-hybrid", "graph-chunk", "parent-child", "all"],
                        default="all",
                        help="which pipeline to run (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of questions per pipeline (useful for smoke tests)")
    args = parser.parse_args()

    from llm.llm_interface import EurecomLLM
    llm = EurecomLLM(
        base_url=os.getenv("EURECOM_LLM_URL"),
        api_key=os.getenv("EURECOM_LLM_KEY"),
        model=os.getenv("EURECOM_LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
    )

    if args.pipeline in ("rag", "all"):
        run_rag_pipeline(llm, limit=args.limit)

    if args.pipeline in ("graphrag", "all"):
        run_graphrag_pipeline(llm, limit=args.limit)

    if args.pipeline in ("hybrid", "all"):
        run_rag_hybrid_pipeline(llm, limit=args.limit)

    if args.pipeline in ("graphrag-hybrid", "all"):
        run_graphrag_hybrid_pipeline(llm, limit=args.limit)

    if args.pipeline in ("graph-chunk", "all"):
        run_graph_chunk_pipeline(llm, limit=args.limit)

    if args.pipeline in ("parent-child", "all"):
        run_parent_child_pipeline(llm, limit=args.limit)
