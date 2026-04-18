from pathlib import Path
import importlib.util
import json
import os
import sys
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
QUESTIONS_DIR = PROJECT_ROOT / "task1_questions_generation" / "output_questions"
OUTPUT_DIR    = Path(__file__).parent / "output"

# ── load sub-modules (graph-rag folder has a hyphen, can't use normal import) ──
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = Path(__file__).parent
sys.path.append(str(_base / "rag"))
sys.path.append(str(_base / "graph-rag"))
sys.path.append(str(_base / "llm"))

run_rag      = _load_module("rag_pipeline",     _base / "rag"       / "pipeline.py").run_rag
run_graphrag = _load_module("graphrag_pipeline", _base / "graph-rag" / "pipeline.py").run_graphrag

from llm_interface import PlaceholderLLM


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

            single_hop = entry.get("graph_single_hop", {})
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

            multi_hop = entry.get("graph_multi_hop", {})
            if multi_hop and multi_hop.get("question") and multi_hop.get("answer"):
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


# ── main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    llm = PlaceholderLLM()

    questions_t5 = load_questions_t5(QUESTIONS_DIR / "questions_t5_lettria.jsonl")
    questions_t5 += load_questions_t5(QUESTIONS_DIR / "questions_t5_oskgc.jsonl")
    print(f"Loaded {len(questions_t5)} T5 questions")

    questions_llm = []
    for fname in ["questions_graph_lettria.jsonl", "questions_graph_oskgc.jsonl"]:
        fpath = QUESTIONS_DIR / fname
        if fpath.exists():
            questions_llm += load_questions_llm(fpath)
    print(f"Loaded {len(questions_llm)} LLM graph questions")

    if not questions_llm:
        print("No LLM graph questions found — skipping pipelines.")
        sys.exit(0)

    all_questions = questions_t5 + questions_llm

    run_and_save(all_questions, run_rag,      llm, OUTPUT_DIR / "rag_results.jsonl",      "RAG")
    run_and_save(all_questions, run_graphrag,  llm, OUTPUT_DIR / "graphrag_results.jsonl", "GraphRAG")
