"""
K-sweep evaluation pipeline: prepare MIRAGE input, run FactCC, then analyze.

Stages:
  prepare  — convert k_sweep JSONL files → CSV inputs for FactCC
  score    — run FactCC on all prepared CSV inputs
  analyze  — load FactCC scores, print table + SBERT similarity vs ground truth
  all      — run all three stages in order (default)

Usage (from project root):
    python task3_evaluation/k_sweep_eval.py --stage all
    python task3_evaluation/k_sweep_eval.py --stage analyze   # just re-print table
    python task3_evaluation/k_sweep_eval.py --stage analyze --plot  # also save a PNG plot
"""

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
SWEEP_DIR      = PROJECT_ROOT / "task2_setup_rag" / "output" / "k_sweep"
INPUTS_DIR     = Path(__file__).parent / "inputs"  / "k_sweep"
OUTPUTS_DIR    = Path(__file__).parent / "outputs" / "k_sweep"
QUESTIONS_BASE = PROJECT_ROOT / "task1_questions_generation" / "output"

# ── Coverage-error patterns (same as prepare_mirage_input.py) ─────────────────
_COVERAGE_ERROR_RE = re.compile("|".join([
    r"i don.?t know", r"i do not know", r"i.?m not sure", r"i am not sure",
    r"i cannot determine", r"i can.?t determine",
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
    r"je ne sais pas", r"le contexte ne (permet|fournit|contient|mentionne|précise)",
    r"aucune information", r"pas (d.?information|assez d.?information)",
    r"(n.?est|ne sont) pas (mentionn|précis|indiqu|fourni)",
    r"impossible de (répondre|déterminer)",
]), re.IGNORECASE)


def is_coverage_error(text) -> bool:
    if not text:
        return True
    s = str(text).strip().replace("’", "'").replace("ʼ", "'")
    return bool(_COVERAGE_ERROR_RE.search(s)) if s else True


# ── Source index (maps question id → original sentence) ───────────────────────
def build_source_index():
    def load_jsonl(p):
        with open(p, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    t5_index, llm_index = {}, {}
    for fname in ("questions_t5_lettria.jsonl", "questions_t5_oskgc.jsonl"):
        p = QUESTIONS_BASE / fname
        if p.exists():
            for e in load_jsonl(p):
                t5_index[e["id"]] = e.get("sentence", "")
    for fname in ("questions_llm_lettria.jsonl", "questions_llm_oskgc.jsonl"):
        p = QUESTIONS_BASE / fname
        if p.exists():
            for e in load_jsonl(p):
                llm_index[e["id"]] = e.get("original_sent", "")
    return t5_index, llm_index


# ── Stage 1: prepare ──────────────────────────────────────────────────────────
def stage_prepare():
    print("\n=== Stage 1: Prepare MIRAGE inputs ===")
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(SWEEP_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"  No JSONL files found in {SWEEP_DIR}")
        print("  Run: python task2_setup_rag/run_k_sweep.py first")
        return

    t5_index, llm_index = build_source_index()

    for jf in jsonl_files:
        csv_out = INPUTS_DIR / (jf.stem + "_input.csv")
        if csv_out.exists():
            print(f"  [skip] {csv_out.name} already exists")
            continue

        with open(jf, encoding="utf-8") as f:
            entries = [json.loads(l) for l in f if l.strip()]

        if not entries:
            print(f"  [skip] {jf.name} is empty")
            continue

        rows = []
        for e in entries:
            hop = e.get("hop_type", "t5")
            source = t5_index.get(e["id"], "") if hop == "t5" else llm_index.get(e["id"], "")
            rows.append({
                "source":            source,
                "gen":               e["llm_answer"],
                "id":                e["id"],
                "question":          e["question"],
                "ground_truth":      e.get("answer", ""),
                "hop_type":          hop,
                "pipeline":          jf.stem,
                "is_coverage_error": is_coverage_error(e["llm_answer"]),
            })

        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        n_cov = sum(1 for r in rows if r["is_coverage_error"])
        print(f"  {jf.name} → {csv_out.name}  ({len(rows)} entries, {n_cov} coverage errors)")

    print("  Prepare done.")


# ── Stage 2: score (FactCC) ───────────────────────────────────────────────────
def stage_score():
    print("\n=== Stage 2: Run FactCC ===")
    sys.path.insert(0, "task3_evaluation/mirage_lib")
    from datasets import load_from_disk
    from mirage import factcc

    csv_files = sorted(INPUTS_DIR.glob("*_input.csv"))
    if not csv_files:
        print(f"  No CSV files found in {INPUTS_DIR}. Run --stage prepare first.")
        return

    for csv_f in csv_files:
        stem = csv_f.stem.replace("_input", "")
        out_dir = OUTPUTS_DIR / stem
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"  [skip] {out_dir.name} already scored")
            continue

        print(f"\n  FactCC on {csv_f.name} …")
        out_dir.mkdir(parents=True, exist_ok=True)

        metric = factcc()
        metric.evaluate_dataset(
            dataset=str(csv_f),
            source_col="source",
            gen_col="gen",
            truncation=True,
            padding=True,
            save_result_dataset_folder_path=str(out_dir),
        )

        # Add faithfulness = 1 - P(halluc)
        ds = load_from_disk(str(out_dir))
        ds = ds.add_column("faithfulness", [1.0 - s for s in ds["score"]])
        tmp = tempfile.mkdtemp(prefix="factcc_swap_", dir=str(out_dir.parent))
        ds.save_to_disk(tmp)
        del ds
        shutil.rmtree(str(out_dir))
        shutil.move(tmp, str(out_dir))

        ds = load_from_disk(str(out_dir))
        print(f"    {ds.num_rows} rows — mean faithfulness: {sum(ds['faithfulness'])/len(ds['faithfulness']):.4f}")

    print("  Score done.")


# ── Stage 3: analyze ──────────────────────────────────────────────────────────
def _parse_stem(stem: str):
    """Extract (pipeline_name, k) from a filename like 'rag_hybrid_k10'."""
    m = re.match(r"^(.+?)_k(\d+)$", stem)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def stage_analyze(plot: bool = False):
    print("\n=== Stage 3: Analysis ===")
    from datasets import load_from_disk

    output_dirs = sorted(OUTPUTS_DIR.glob("*_k*"))
    if not output_dirs:
        print(f"  No scored outputs found in {OUTPUTS_DIR}. Run --stage score first.")
        return

    # ── Collect per (pipeline, k) stats ───────────────────────────────────────
    # Structure: results[pipeline][k] = {faithfulness: [...], coverage_errors: int, total: int}
    results = defaultdict(lambda: defaultdict(lambda: {"faithfulness": [], "cov_err": 0, "total": 0}))
    # Also per hop_type: hop_results[pipeline][k][hop_type] = [faithfulness]
    hop_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for d in output_dirs:
        pipeline_name, k = _parse_stem(d.name)
        if pipeline_name is None:
            continue
        try:
            ds = load_from_disk(str(d))
        except Exception as e:
            print(f"  [warn] could not load {d.name}: {e}")
            continue

        # Read coverage error flags from the CSV (not stored in FactCC output)
        csv_path = INPUTS_DIR / (d.name + "_input.csv")
        cov_flags = {}
        if csv_path.exists():
            with open(csv_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    key = (row["id"], row.get("hop_type", "t5"))
                    cov_flags[key] = row.get("is_coverage_error", "False") == "True"

        has_hop_type = "hop_type" in ds.column_names
        for row in ds:
            hop = row.get("hop_type", "t5") if has_hop_type else "unknown"
            key = (str(row.get("id", "")), hop)
            is_cov = cov_flags.get(key, False)

            results[pipeline_name][k]["total"] += 1
            if is_cov:
                results[pipeline_name][k]["cov_err"] += 1
            else:
                f_score = row["faithfulness"]
                results[pipeline_name][k]["faithfulness"].append(f_score)
                hop_results[pipeline_name][k][hop].append(f_score)

    if not results:
        print("  No results to analyze.")
        return

    # ── Print summary table ────────────────────────────────────────────────────
    pipelines = sorted(results.keys())
    all_k = sorted({k for p in results for k in results[p]})

    print(f"\n{'Pipeline':<20} | " + " | ".join(f"K={k:>2}" for k in all_k))
    print("-" * (22 + 9 * len(all_k)))

    best_per_pipeline = {}
    for pipeline in pipelines:
        row_vals = []
        best_k, best_f = None, -1
        for k in all_k:
            data = results[pipeline].get(k)
            if data and data["faithfulness"]:
                mean_f = sum(data["faithfulness"]) / len(data["faithfulness"])
                row_vals.append(f"{mean_f:.4f}")
                if mean_f > best_f:
                    best_f, best_k = mean_f, k
            else:
                row_vals.append("  —   ")
        best_per_pipeline[pipeline] = (best_k, best_f)
        print(f"{pipeline:<20} | " + " | ".join(row_vals))

    print()
    print("Coverage error rate (abstentions, excluded from faithfulness):")
    print(f"{'Pipeline':<20} | " + " | ".join(f"K={k:>2}" for k in all_k))
    print("-" * (22 + 9 * len(all_k)))
    for pipeline in pipelines:
        row_vals = []
        for k in all_k:
            data = results[pipeline].get(k)
            if data and data["total"]:
                rate = data["cov_err"] / data["total"]
                row_vals.append(f"{rate:.1%}  ")
            else:
                row_vals.append("  —   ")
        print(f"{pipeline:<20} | " + " | ".join(row_vals))

    print()
    print("Best K per pipeline:")
    for pipeline in pipelines:
        best_k, best_f = best_per_pipeline[pipeline]
        print(f"  {pipeline:<22} → K={best_k}  (faithfulness={best_f:.4f})")

    # ── Per hop_type breakdown ─────────────────────────────────────────────────
    hop_types = sorted({h for p in hop_results for k in hop_results[p] for h in hop_results[p][k]})
    if hop_types:
        print()
        print("Faithfulness by hop_type (mean across K values):")
        header = f"{'Pipeline':<20} | " + " | ".join(f"{h:<10}" for h in hop_types)
        print(header)
        print("-" * len(header))
        for pipeline in pipelines:
            hop_avgs = []
            for ht in hop_types:
                scores = [
                    s
                    for k in hop_results[pipeline]
                    for s in hop_results[pipeline][k].get(ht, [])
                ]
                hop_avgs.append(f"{sum(scores)/len(scores):.4f}    " if scores else "  —       ")
            print(f"{pipeline:<20} | " + " | ".join(hop_avgs))

    # ── Optional plot ──────────────────────────────────────────────────────────
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            markers = ["o", "s", "^", "D", "v"]
            for i, pipeline in enumerate(pipelines):
                xs, ys = [], []
                for k in all_k:
                    data = results[pipeline].get(k)
                    if data and data["faithfulness"]:
                        xs.append(k)
                        ys.append(sum(data["faithfulness"]) / len(data["faithfulness"]))
                if xs:
                    ax.plot(xs, ys, marker=markers[i % len(markers)], label=pipeline)

            ax.set_xlabel("K (number of retrieved items)")
            ax.set_ylabel("Mean faithfulness (FactCC, excluding abstentions)")
            ax.set_title("Faithfulness vs K — all pipelines")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_path = OUTPUTS_DIR / "k_sweep_faithfulness.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"\nPlot saved to {plot_path}")
        except ImportError:
            print("\n  [info] matplotlib not available — skipping plot")

    print("\nAnalysis done.")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-sweep evaluation: prepare, score, analyze.")
    parser.add_argument("--stage",
                        choices=["prepare", "score", "analyze", "all"],
                        default="all")
    parser.add_argument("--plot", action="store_true",
                        help="Save a faithfulness-vs-K plot (PNG) during analyze stage")
    args = parser.parse_args()

    if args.stage in ("prepare", "all"):
        stage_prepare()
    if args.stage in ("score", "all"):
        stage_score()
    if args.stage in ("analyze", "all"):
        stage_analyze(plot=args.plot)
