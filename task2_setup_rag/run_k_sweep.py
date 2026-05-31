"""
K-sweep experiment: run each pipeline with multiple K values to find the optimal
number of retrieved items per pipeline type.

Usage:
    # Run all pipelines with K ∈ {2,5,10,15,20}, 50 questions each
    python task2_setup_rag/run_k_sweep.py

    # Custom K values and question limit
    python task2_setup_rag/run_k_sweep.py --pipeline rag --k_values 3 7 12 --limit 30

    # All pipelines, full question set
    python task2_setup_rag/run_k_sweep.py --pipeline all --k_values 2 5 10 15 20

Outputs: task2_setup_rag/output/k_sweep/{pipeline}_k{k}.jsonl
         (resume-safe: already-answered questions are skipped)
"""

from pathlib import Path
from dotenv import load_dotenv
import argparse
import json
import os
import sys
import time
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
QUESTIONS_DIR = PROJECT_ROOT / "task1_questions_generation" / "output"
SWEEP_DIR     = Path(__file__).parent / "output" / "k_sweep"

load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_K_VALUES = [2, 5, 10, 15, 20]
ALL_PIPELINES    = ["rag", "graphrag", "rag-hybrid", "graphrag-hybrid", "graph-chunk"]


# ── Retry / JSON helpers (same as run_pipelines.py) ───────────────────────────
def _call_with_retry(fn, *args, **kwargs):
    delay = 30
    delay_max = 300
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
            print(f"  → gateway likely offline. Sleeping {delay}s then retrying…")
            try:
                time.sleep(delay)
            except KeyboardInterrupt:
                raise
            delay = min(delay * 2, delay_max)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ── Question loaders (same as run_pipelines.py) ────────────────────────────────
def load_questions_t5(file_path):
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            for qa in entry["qa_pairs"]:
                if qa["question"] and qa["answer"]:
                    questions.append({
                        "id": entry["id"], "source_id": entry["source_id"],
                        "dataset": entry["dataset"], "category": entry["category"],
                        "hop_type": "t5", "question": qa["question"], "answer": qa["answer"],
                    })
    return questions


def load_questions_llm(file_path):
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            for hop_key, hop_type in [("graph_single_hop", "single_hop"), ("graph_multi_hop", "multi_hop")]:
                hop = entry.get(hop_key) or {}
                if hop.get("question") and hop.get("answer"):
                    q = {
                        "id": entry["id"], "source_id": entry["source_id"],
                        "dataset": entry["dataset"], "category": entry["category"],
                        "hop_type": hop_type, "question": hop["question"], "answer": hop["answer"],
                    }
                    if hop_key == "graph_multi_hop":
                        q["chain"] = hop.get("chain")
                    questions.append(q)
    return questions


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


# ── Generic runner ─────────────────────────────────────────────────────────────
def run_and_save(questions, pipeline_fn, llm, output_file, label):
    os.makedirs(output_file.parent, exist_ok=True)

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
            print(f"[{label}] [{i+1}/{len(questions)}] {q['id']} ({q['hop_type']})")
            output = _call_with_retry(pipeline_fn, q["question"], llm)
            record = {
                "id": q["id"], "source_id": q["source_id"], "dataset": q["dataset"],
                "category": q["category"], "hop_type": q["hop_type"],
                "question": q["question"], "answer": q["answer"],
                **output
            }
            f.write(json.dumps(record, ensure_ascii=False, cls=_NumpyEncoder) + "\n")
            f.flush()
            written += 1

    print(f"  Wrote {written} new results to {output_file.name} (total: {len(done) + written})")


# ── Per-pipeline sweep runners ─────────────────────────────────────────────────
def sweep_rag(questions, llm, k_values):
    from rag.pipeline import run_rag
    for k in k_values:
        out = SWEEP_DIR / f"rag_k{k}.jsonl"
        run_and_save(questions, lambda q, l, _k=k: run_rag(q, l, k=_k), llm, out, f"RAG k={k}")


def sweep_graphrag(questions, llm, k_values):
    from graph_rag.pipeline import run_graphrag
    for k in k_values:
        out = SWEEP_DIR / f"graphrag_k{k}.jsonl"
        run_and_save(questions, lambda q, l, _k=k: run_graphrag(q, l, k=_k), llm, out, f"GraphRAG k={k}")


def sweep_rag_hybrid(questions, llm, k_values):
    from rag.pipeline_hybrid import run_rag_hybrid
    for k in k_values:
        out = SWEEP_DIR / f"rag_hybrid_k{k}.jsonl"
        run_and_save(questions, lambda q, l, _k=k: run_rag_hybrid(q, l, k=_k), llm, out, f"RAG-Hybrid k={k}")


def sweep_graphrag_hybrid(questions, llm, k_values):
    from graph_rag.pipeline_hybrid import run_graphrag_hybrid
    for k in k_values:
        out = SWEEP_DIR / f"graphrag_hybrid_k{k}.jsonl"
        run_and_save(
            questions, lambda q, l, _k=k: run_graphrag_hybrid(q, l, k=_k),
            llm, out, f"GraphRAG-Hybrid k={k}"
        )


def sweep_graph_chunk(questions, llm, k_values):
    """For graph-chunk, test with k_graph=k_chunk=k (total context = 2k items)."""
    from graph_rag.pipeline_graph_chunk import run_graph_chunk
    for k in k_values:
        out = SWEEP_DIR / f"graph_chunk_k{k}.jsonl"
        run_and_save(
            questions, lambda q, l, _k=k: run_graph_chunk(q, l, k_graph=_k, k_chunk=_k),
            llm, out, f"Graph-Chunk k={k}"
        )


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-sweep: find optimal K for each pipeline type.")
    parser.add_argument("--pipeline",
                        choices=["rag", "graphrag", "rag-hybrid", "graphrag-hybrid", "graph-chunk", "all"],
                        default="all")
    parser.add_argument("--k_values", type=int, nargs="+", default=DEFAULT_K_VALUES,
                        help=f"K values to test (default: {DEFAULT_K_VALUES})")
    parser.add_argument("--limit", type=int, default=50,
                        help="max number of questions per (pipeline, K) run (default: 50)")
    args = parser.parse_args()

    k_values = sorted(set(args.k_values))
    print(f"K sweep — pipelines: {args.pipeline}  K values: {k_values}  limit: {args.limit}")
    print(f"Output: {SWEEP_DIR}")

    questions = load_all_questions()
    print(f"Loaded {len(questions)} questions total")
    if args.limit:
        questions = questions[:args.limit]
        print(f"  → capped to {len(questions)} for this sweep")

    from llm.llm_interface import EurecomLLM
    llm = EurecomLLM(
        base_url=os.getenv("EURECOM_LLM_URL"),
        api_key=os.getenv("EURECOM_LLM_KEY"),
        model=os.getenv("EURECOM_LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
    )

    if args.pipeline in ("rag", "all"):
        print("\n=== RAG ===")
        sweep_rag(questions, llm, k_values)

    if args.pipeline in ("graphrag", "all"):
        print("\n=== GraphRAG ===")
        sweep_graphrag(questions, llm, k_values)

    if args.pipeline in ("rag-hybrid", "all"):
        print("\n=== RAG Hybrid ===")
        sweep_rag_hybrid(questions, llm, k_values)

    if args.pipeline in ("graphrag-hybrid", "all"):
        print("\n=== GraphRAG Hybrid ===")
        sweep_graphrag_hybrid(questions, llm, k_values)

    if args.pipeline in ("graph-chunk", "all"):
        print("\n=== Graph-Chunk ===")
        sweep_graph_chunk(questions, llm, k_values)

    print("\nSweep complete. Run task3_evaluation/k_sweep_eval.py to evaluate results.")
