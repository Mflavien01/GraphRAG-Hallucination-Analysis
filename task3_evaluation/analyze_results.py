"""  
Task 3 - Comparative analysis of FactCC scores.
FactCC score: 1 = consistent with context / 0 = hallucination
prediction: 1 = CORRECT / 0 = HALLUCINATED (according to the model)
We compare RAG vs GraphRAG on 3 axes:
1. Overall score
2. By hop_type (t5 / single_hop / multi_hop)
3. "Hallucination" rate (prediction == 0)

"""

import pandas as pd
from datasets import load_from_disk
from pathlib import Path


def load_scores(output_dir: str) -> pd.DataFrame:
    """Load the HuggingFace dataset produced by MIRAGE."""
    ds = load_from_disk(output_dir)
    return ds.to_pandas()


def print_section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


def analyze():
    rag_df  = load_scores("task3_evaluation/outputs/factcc_rag")
    grag_df = load_scores("task3_evaluation/outputs/factcc_graphrag")

    # ── 1. Global Score ──────────────────────────────────
    print_section("GLOBAL SCORE")
    summary = pd.DataFrame({
        "mean_score":       [rag_df["score"].mean(),       grag_df["score"].mean()],
        "std_score":        [rag_df["score"].std(),        grag_df["score"].std()],
        "halluc_rate": [(rag_df["score"] < 0.5).mean(),
                (grag_df["score"] < 0.5).mean()],
    }, index=["RAG", "GraphRAG"])
    print(summary.round(4))

    # ── 2. By hop_type ────────────────────────────────────
    print_section("BY HOP_TYPE — AVERAGE SCORE")
    for name, df in [("RAG", rag_df), ("GraphRAG", grag_df)]:
        print(f"\n{name}:")
        g = df.groupby("hop_type").agg(
            mean_score=("score", "mean"),
            halluc_rate=("score", lambda x: (x < 0.5).mean()),
            count=("score", "count")
        ).round(4)
        print(g)

    # ── 3. Comparison by hop_type  ────────────
    print_section("DELTA GraphRAG - RAG (by hop_type)")
    rag_hop  = rag_df.groupby("hop_type")["score"].mean()
    grag_hop = grag_df.groupby("hop_type")["score"].mean()
    delta = (grag_hop - rag_hop).rename("delta_score").round(4)
    print(delta)
    print("\n(negative = GraphRAG more hallucinatory than RAG on this type)")

    # ── 4. Critical cases: very low score (<0.3) ───────────
    print_section("CRITICAL CASES (score < 0.3)")
    for name, df in [("RAG", rag_df), ("GraphRAG", grag_df)]:
        bad = df[df["score"] < 0.3]
        print(f"\n{name}: {len(bad)}/{len(df)} critical entries")
        if len(bad):
            print(bad[["id","hop_type","question","gen","score"]].head(5).to_string(index=False))

    # ── 5. CSV export for the report ─────────────────────
    out = Path("task3_evaluation/outputs/")
    rag_df.to_csv(out / "factcc_rag_scores.csv", index=False)
    grag_df.to_csv(out / "factcc_graphrag_scores.csv", index=False)
    print(f"\nExported → {out}")


    # ── 6. Numeric vs text analysis ─────────────────────
    import re
    def is_numeric_answer(text):
        return bool(re.match(r'^[\d\s\.,\-\+]+(%|AU|km|m|min|minutes|members)?$',
                             str(text).strip()))

    for name, df in [("RAG", rag_df), ("GraphRAG", grag_df)]:
        df["is_numeric"] = df["gen"].apply(is_numeric_answer)
        num = df[df["is_numeric"]]
        txt = df[~df["is_numeric"]]
        print(f"\n{name} — Numeric ({len(num)} entries): score={num['score'].mean():.4f}")
        print(f"{name} — Text    ({len(txt)} entries): score={txt['score'].mean():.4f}")

    # ── 7. Score distribution (for the report) ──────
    print_section("SCORE DISTRIBUTION")
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        rag_pct  = (rag_df["score"]  >= threshold).mean()
        grag_pct = (grag_df["score"] >= threshold).mean()
        print(f"score >= {threshold}: RAG={rag_pct:.1%}  GraphRAG={grag_pct:.1%}")


def compare_hybrid():
    """Compare RAG baseline vs RAG hybrid pour mesurer l'impact du BM25."""
    print_section("RAG BASELINE vs RAG HYBRID (BM25)")

    baseline = load_scores("task3_evaluation/outputs/factcc_rag")
    hybrid   = load_scores("task3_evaluation/outputs/factcc_rag_hybrid")

    print(f"\nBaseline RAG : score={baseline['score'].mean():.4f}  halluc={( baseline['score'] < 0.5).mean():.1%}")
    print(f"Hybrid  RAG  : score={hybrid['score'].mean():.4f}  halluc={(hybrid['score'] < 0.5).mean():.1%}")
    print(f"Delta        : {hybrid['score'].mean() - baseline['score'].mean():+.4f}")

    print("\n--- Par hop_type ---")
    for hop in ['t5', 'single_hop', 'multi_hop']:
        b = baseline[baseline['hop_type'] == hop]['score'].mean()
        h = hybrid[hybrid['hop_type'] == hop]['score'].mean()
        print(f"  {hop:12s}  baseline={b:.4f}  hybrid={h:.4f}  Δ={h-b:+.4f}")
        
        
if __name__ == "__main__":
    analyze()
    compare_hybrid()