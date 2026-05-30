"""
Task 3 - Comparative analysis of FactCC scores.

MIRAGE writes `score = P(INCORRECT) = P(hallucination)`.
This script analyses the inverse `faithfulness = P(CORRECT) = 1 - score`, so
all metrics use the natural "higher = better" convention.

We compare RAG vs GraphRAG on:
1. Overall faithfulness (mean, std, hallucination rate at 0.5 threshold)
2. By hop_type (t5 / single_hop / multi_hop)
3. Critical cases (faithfulness < 0.3)
4. Faithfulness distribution by threshold
5. SBERT cosine similarity (gen vs ground_truth)
6. Sanity check: Pearson(faithfulness, sbert_cosine) — must be positive
"""

import numpy as np
import pandas as pd
from datasets import load_from_disk
from pathlib import Path
from sentence_transformers import SentenceTransformer

from prepare_mirage_input import is_coverage_error


def load_scores(output_dir: str) -> pd.DataFrame:
    """Load the HuggingFace dataset produced by MIRAGE.
    Adds `faithfulness = 1 - score` if not already present on disk.
    Adds `is_coverage_error` (recomputed from `gen`) if absent, so outputs
    generated before the abstention-flag change are still handled correctly."""
    ds = load_from_disk(output_dir)
    df = ds.to_pandas()
    if "faithfulness" not in df.columns:
        df["faithfulness"] = 1.0 - df["score"]
    if "is_coverage_error" not in df.columns:
        df["is_coverage_error"] = df["gen"].apply(is_coverage_error)
    else:
        df["is_coverage_error"] = df["is_coverage_error"].astype(bool)
    return df


def split_coverage(df: pd.DataFrame):
    """Split into (answered, coverage_errors). Faithfulness metrics use `answered`
    only; abstentions are reported separately as a coverage-error rate."""
    answered = df[~df["is_coverage_error"]].copy()
    coverage = df[df["is_coverage_error"]].copy()
    return answered, coverage


def print_section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


def add_sbert_cosine(df: pd.DataFrame, model: SentenceTransformer) -> pd.DataFrame:
    """
    Add a 'sbert_cosine' column: cosine similarity between 'gen' and 'ground_truth'.
    - Rows where either field is empty / None receive None (excluded from averages).
    - All texts are encoded in a single model.encode() batch pass.
    - normalize_embeddings=True lets us use a cheap dot-product as cosine sim.
    Returns a copy of df with the new column appended.
    """
    gens = df["gen"].fillna("").astype(str).tolist()
    gts  = df["ground_truth"].fillna("").astype(str).tolist()

    # Keep only rows where both sides are non-empty
    valid_idx = [i for i, (g, gt) in enumerate(zip(gens, gts))
                 if g.strip() and gt.strip()]

    scores: list = [None] * len(df)

    if valid_idx:
        gen_texts = [gens[i] for i in valid_idx]
        gt_texts  = [gts[i]  for i in valid_idx]

        # One encode call for all texts: gen texts first, then gt texts
        all_embs = model.encode(
            gen_texts + gt_texts,
            batch_size=64,
            normalize_embeddings=True,   # unit vectors → dot product == cosine sim
            show_progress_bar=False,
        )
        gen_embs = all_embs[:len(valid_idx)]  # shape (N, d)
        gt_embs  = all_embs[len(valid_idx):]  # shape (N, d)

        # Row-wise dot product of unit vectors = cosine similarity
        sims = (gen_embs * gt_embs).sum(axis=1)  # shape (N,)

        for rank, i in enumerate(valid_idx):
            scores[i] = float(sims[rank])

    out = df.copy()
    out["sbert_cosine"] = scores
    return out


def analyze(model: SentenceTransformer):
    rag_df  = load_scores("task3_evaluation/outputs/factcc_rag")
    grag_df = load_scores("task3_evaluation/outputs/factcc_graphrag")

    # Compute SBERT cosine similarity once, before any display section.
    # sbert_cosine is then present in the df for every section below (incl. CSV export).
    rag_df  = add_sbert_cosine(rag_df,  model)
    grag_df = add_sbert_cosine(grag_df, model)

    # ── 0. Coverage errors (abstentions) ─────────────────
    # Rows where the model declined to answer for lack of context. FactCC still
    # scored them, but an abstention is neither faithful nor hallucinated, so we
    # report it separately and EXCLUDE it from every faithfulness statistic below.
    print_section("COVERAGE ERROR RATE (abstentions, excluded from faithfulness)")
    cov_summary = pd.DataFrame({
        "n_total":          [len(rag_df),                       len(grag_df)],
        "n_coverage_error": [int(rag_df["is_coverage_error"].sum()),
                             int(grag_df["is_coverage_error"].sum())],
        "coverage_error_rate": [rag_df["is_coverage_error"].mean(),
                                grag_df["is_coverage_error"].mean()],
    }, index=["RAG", "GraphRAG"])
    print(cov_summary.round(4))

    rag_ans,  _ = split_coverage(rag_df)
    grag_ans, _ = split_coverage(grag_df)

    # ── 1. Global Faithfulness (answered rows only) ──────
    print_section("GLOBAL FAITHFULNESS (coverage errors excluded)")
    summary = pd.DataFrame({
        "n_answered":        [len(rag_ans),                len(grag_ans)],
        "mean_faithfulness": [rag_ans["faithfulness"].mean(),  grag_ans["faithfulness"].mean()],
        "std_faithfulness":  [rag_ans["faithfulness"].std(),   grag_ans["faithfulness"].std()],
        "halluc_rate":       [(rag_ans["faithfulness"] < 0.5).mean(),
                              (grag_ans["faithfulness"] < 0.5).mean()],
    }, index=["RAG", "GraphRAG"])
    print(summary.round(4))

    # ── 2. By hop_type ────────────────────────────────────
    print_section("BY HOP_TYPE — MEAN FAITHFULNESS (coverage errors excluded)")
    for name, df in [("RAG", rag_ans), ("GraphRAG", grag_ans)]:
        print(f"\n{name}:")
        g = df.groupby("hop_type").agg(
            mean_faithfulness=("faithfulness", "mean"),
            halluc_rate=("faithfulness", lambda x: (x < 0.5).mean()),
            count=("faithfulness", "count")
        ).round(4)
        print(g)

    # ── 3. Comparison by hop_type ─────────────────────────
    print_section("DELTA GraphRAG - RAG (by hop_type)")
    rag_hop  = rag_ans.groupby("hop_type")["faithfulness"].mean()
    grag_hop = grag_ans.groupby("hop_type")["faithfulness"].mean()
    delta = (grag_hop - rag_hop).rename("delta_faithfulness").round(4)
    print(delta)
    print("\n(positive = GraphRAG more faithful than RAG on this type)")

    # ── 4. Critical cases: very low faithfulness (<0.3) ──
    print_section("CRITICAL CASES (faithfulness < 0.3, coverage errors excluded)")
    for name, df in [("RAG", rag_ans), ("GraphRAG", grag_ans)]:
        bad = df[df["faithfulness"] < 0.3]
        print(f"\n{name}: {len(bad)}/{len(df)} critical entries")
        if len(bad):
            print(bad[["id","hop_type","question","gen","faithfulness"]].head(5).to_string(index=False))

    # ── 5. CSV export for the report ─────────────────────
    out = Path("task3_evaluation/outputs/")
    rag_df.to_csv(out / "factcc_rag_scores.csv", index=False)
    grag_df.to_csv(out / "factcc_graphrag_scores.csv", index=False)
    print(f"\nExported → {out}")


    # ── 6. Numeric vs text analysis ─────────────────────
    import re
    _NUM_PAT = re.compile(
        r'^[-+]?[\d,\.]*\d[\d,\.]*'   # numeric core: at least one digit
        r'(\s+[(]?[a-zA-Z]+[)]?)*'    # optional unit word(s), possibly parenthesised
        r'$'
    )

    def is_numeric_answer(text):
        """True if ground_truth is a numeric quantity (possibly with unit words)."""
        s = str(text).strip()
        return bool(s and re.search(r'\d', s) and _NUM_PAT.match(s))

    for name, df in [("RAG", rag_ans), ("GraphRAG", grag_ans)]:
        df["gt_is_numeric"] = df["ground_truth"].apply(is_numeric_answer)
        num = df[df["gt_is_numeric"]]
        txt = df[~df["gt_is_numeric"]]
        print(f"\n{name} — Numeric gt ({len(num)} entries): faithfulness={num['faithfulness'].mean():.4f}")
        print(f"{name} — Text    gt ({len(txt)} entries): faithfulness={txt['faithfulness'].mean():.4f}")

    # ── 7. Faithfulness distribution (for the report) ──────
    print_section("FAITHFULNESS DISTRIBUTION (coverage errors excluded)")
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        rag_pct  = (rag_ans["faithfulness"]  >= threshold).mean()
        grag_pct = (grag_ans["faithfulness"] >= threshold).mean()
        print(f"faithfulness >= {threshold}: RAG={rag_pct:.1%}  GraphRAG={grag_pct:.1%}")

    # ── 8. Sentence-BERT Cosine Similarity ───────────────────
    print_section("SENTENCE-BERT COSINE SIMILARITY  (gen ↔ ground_truth, coverage errors excluded)")
    sbert_summary = pd.DataFrame({
        "mean_sbert": [rag_ans["sbert_cosine"].mean(),        grag_ans["sbert_cosine"].mean()],
        "std_sbert":  [rag_ans["sbert_cosine"].std(),         grag_ans["sbert_cosine"].std()],
        "n_valid":    [rag_ans["sbert_cosine"].notna().sum(),  grag_ans["sbert_cosine"].notna().sum()],
    }, index=["RAG", "GraphRAG"])
    print(sbert_summary.round(4))

    print_section("SENTENCE-BERT — BY HOP_TYPE")
    for name, df in [("RAG", rag_ans), ("GraphRAG", grag_ans)]:
        print(f"\n{name}:")
        g = df.groupby("hop_type")["sbert_cosine"].agg(
            mean_sbert="mean",
            std_sbert="std",
            n_valid="count",
        ).round(4)
        print(g)

    # ── 9. Sanity check: correlation faithfulness ↔ SBERT ──
    # If faithfulness is correctly oriented (higher = better), the Pearson
    # correlation with SBERT(gen, ground_truth) must be POSITIVE.
    # With the pre-fix interpretation (score = P(halluc) misread as P(faithful))
    # this correlation was about −0.33.
    print_section("SANITY CHECK — Pearson(faithfulness, sbert_cosine)")
    for name, df in [("RAG", rag_ans), ("GraphRAG", grag_ans)]:
        sub = df.dropna(subset=["sbert_cosine"])
        r = sub["faithfulness"].corr(sub["sbert_cosine"])
        print(f"  {name:8s}  r = {r:+.4f}   (n={len(sub)})")


def compare_hybrid(model: SentenceTransformer):
    """Compare RAG baseline vs RAG hybrid pour mesurer l'impact du BM25."""
    print_section("RAG BASELINE vs RAG HYBRID (BM25)")

    baseline = load_scores("task3_evaluation/outputs/factcc_rag")
    hybrid   = load_scores("task3_evaluation/outputs/factcc_rag_hybrid")

    baseline = add_sbert_cosine(baseline, model)
    hybrid   = add_sbert_cosine(hybrid,   model)

    print(f"\nCoverage errors (excluded): baseline={baseline['is_coverage_error'].sum()}/{len(baseline)} "
          f"({baseline['is_coverage_error'].mean():.1%})  "
          f"hybrid={hybrid['is_coverage_error'].sum()}/{len(hybrid)} "
          f"({hybrid['is_coverage_error'].mean():.1%})")

    baseline, _ = split_coverage(baseline)
    hybrid,   _ = split_coverage(hybrid)

    print(f"\nBaseline RAG : faithfulness={baseline['faithfulness'].mean():.4f}  halluc={(baseline['faithfulness'] < 0.5).mean():.1%}")
    print(f"Hybrid  RAG  : faithfulness={hybrid['faithfulness'].mean():.4f}  halluc={(hybrid['faithfulness']  < 0.5).mean():.1%}")
    print(f"Delta        : {hybrid['faithfulness'].mean() - baseline['faithfulness'].mean():+.4f}")

    print("\n--- Par hop_type ---")
    for hop in ['t5', 'single_hop', 'multi_hop']:
        b = baseline[baseline['hop_type'] == hop]['faithfulness'].mean()
        h = hybrid[hybrid['hop_type'] == hop]['faithfulness'].mean()
        print(f"  {hop:12s}  baseline={b:.4f}  hybrid={h:.4f}  Δ={h-b:+.4f}")

    # ── SENTENCE-BERT — baseline vs hybrid ──────────────────
    print_section("SENTENCE-BERT — RAG BASELINE vs RAG HYBRID")
    print(f"\nBaseline RAG : sbert={baseline['sbert_cosine'].mean():.4f}  std={baseline['sbert_cosine'].std():.4f}")
    print(f"Hybrid  RAG  : sbert={hybrid['sbert_cosine'].mean():.4f}  std={hybrid['sbert_cosine'].std():.4f}")
    print(f"Delta        : {hybrid['sbert_cosine'].mean() - baseline['sbert_cosine'].mean():+.4f}")

    print("\n--- Par hop_type (SBERT) ---")
    for hop in ['t5', 'single_hop', 'multi_hop']:
        b = baseline[baseline['hop_type'] == hop]['sbert_cosine'].mean()
        h = hybrid[hybrid['hop_type'] == hop]['sbert_cosine'].mean()
        print(f"  {hop:12s}  baseline={b:.4f}  hybrid={h:.4f}  Δ={h-b:+.4f}")
        
        
if __name__ == "__main__":
    print("Loading SBERT model (all-MiniLM-L6-v2) …")
    _sbert = SentenceTransformer("all-MiniLM-L6-v2")
    analyze(_sbert)
    compare_hybrid(_sbert)