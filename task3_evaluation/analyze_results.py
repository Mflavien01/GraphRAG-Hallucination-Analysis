"""  
Task 3 - Comparative analysis of FactCC scores.
FactCC score: 1 = consistent with context / 0 = hallucination
prediction: 1 = CORRECT / 0 = HALLUCINATED (according to the model)
We compare RAG vs GraphRAG on 3 axes:
1. Overall score
2. By hop_type (t5 / single_hop / multi_hop)
3. "Hallucination" rate (prediction == 0)

"""

import numpy as np  # noqa: F401 — used implicitly via SBERT numpy arrays
import pandas as pd
from datasets import load_from_disk
from pathlib import Path
from sentence_transformers import SentenceTransformer


def load_scores(output_dir: str) -> pd.DataFrame:
    """Load the HuggingFace dataset produced by MIRAGE."""
    ds = load_from_disk(output_dir)
    return ds.to_pandas()


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
    _NUM_PAT = re.compile(
        r'^[-+]?[\d,\.]*\d[\d,\.]*'   # numeric core: at least one digit
        r'(\s+[(]?[a-zA-Z]+[)]?)*'    # optional unit word(s), possibly parenthesised
        r'$'
    )

    def is_numeric_answer(text):
        """True if ground_truth is a numeric quantity (possibly with unit words)."""
        s = str(text).strip()
        return bool(s and re.search(r'\d', s) and _NUM_PAT.match(s))

    for name, df in [("RAG", rag_df), ("GraphRAG", grag_df)]:
        df["gt_is_numeric"] = df["ground_truth"].apply(is_numeric_answer)
        num = df[df["gt_is_numeric"]]
        txt = df[~df["gt_is_numeric"]]
        print(f"\n{name} — Numeric gt ({len(num)} entries): score={num['score'].mean():.4f}")
        print(f"{name} — Text    gt ({len(txt)} entries): score={txt['score'].mean():.4f}")

    # ── 7. Score distribution (for the report) ──────
    print_section("SCORE DISTRIBUTION")
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        rag_pct  = (rag_df["score"]  >= threshold).mean()
        grag_pct = (grag_df["score"] >= threshold).mean()
        print(f"score >= {threshold}: RAG={rag_pct:.1%}  GraphRAG={grag_pct:.1%}")

    # ── 8. Sentence-BERT Cosine Similarity ───────────────────
    print_section("SENTENCE-BERT COSINE SIMILARITY  (gen ↔ ground_truth)")
    sbert_summary = pd.DataFrame({
        "mean_sbert": [rag_df["sbert_cosine"].mean(),        grag_df["sbert_cosine"].mean()],
        "std_sbert":  [rag_df["sbert_cosine"].std(),         grag_df["sbert_cosine"].std()],
        "n_valid":    [rag_df["sbert_cosine"].notna().sum(),  grag_df["sbert_cosine"].notna().sum()],
    }, index=["RAG", "GraphRAG"])
    print(sbert_summary.round(4))

    print_section("SENTENCE-BERT — BY HOP_TYPE")
    for name, df in [("RAG", rag_df), ("GraphRAG", grag_df)]:
        print(f"\n{name}:")
        g = df.groupby("hop_type")["sbert_cosine"].agg(
            mean_sbert="mean",
            std_sbert="std",
            n_valid="count",
        ).round(4)
        print(g)


def compare_hybrid(model: SentenceTransformer):
    """Compare RAG baseline vs RAG hybrid pour mesurer l'impact du BM25."""
    print_section("RAG BASELINE vs RAG HYBRID (BM25)")

    baseline = load_scores("task3_evaluation/outputs/factcc_rag")
    hybrid   = load_scores("task3_evaluation/outputs/factcc_rag_hybrid")

    baseline = add_sbert_cosine(baseline, model)
    hybrid   = add_sbert_cosine(hybrid,   model)

    print(f"\nBaseline RAG : score={baseline['score'].mean():.4f}  halluc={( baseline['score'] < 0.5).mean():.1%}")
    print(f"Hybrid  RAG  : score={hybrid['score'].mean():.4f}  halluc={(hybrid['score'] < 0.5).mean():.1%}")
    print(f"Delta        : {hybrid['score'].mean() - baseline['score'].mean():+.4f}")

    print("\n--- Par hop_type ---")
    for hop in ['t5', 'single_hop', 'multi_hop']:
        b = baseline[baseline['hop_type'] == hop]['score'].mean()
        h = hybrid[hybrid['hop_type'] == hop]['score'].mean()
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