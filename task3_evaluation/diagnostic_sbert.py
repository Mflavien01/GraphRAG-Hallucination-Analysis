#!/usr/bin/env python3
"""
diagnostic_sbert.py — Diagnostic de la metrique SBERT cosine similarity (Task 3)

A lancer depuis la racine du repo, comme analyze_results.py :
    python task3_evaluation/diagnostic_sbert.py

Questions auxquelles ce script repond :

  1) VALIDITE DE LA METRIQUE SBERT
     Les scores SBERT sont bas partout (~0.34-0.48). Artefact de format
     (gen = phrase complete vs ground_truth = span court) ou vrai signal ?
     -> Diagnostic A (longueurs) + Diagnostic B (echantillons).

  2) COMPLEMENTARITE DES METRIQUES
     FactCC (fidelite) et SBERT (correction) mesurent-ils la meme chose ?
     -> Diagnostic C : correlation FactCC <-> SBERT.

  3) FactCC EST-IL FIABLE ?  (nouveau)
     Certaines reponses manifestement correctes sont scorees FactCC ~0.
     Est-ce que (a) FactCC se trompe sur ce domaine, ou (b) le champ
     `source` indexe n'est pas la bonne phrase (bug de routage) ?
     -> Diagnostic D : on isole les cas "SBERT haut + FactCC bas" et on
        IMPRIME LE CHAMP source pour trancher.

Aucune dependance nouvelle : numpy, pandas, datasets, sentence-transformers.
"""

import numpy as np
import pandas as pd
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
FACTCC_DIRS = {
    "RAG":      "task3_evaluation/outputs/factcc_rag",
    "GraphRAG": "task3_evaluation/outputs/factcc_graphrag",
}
SBERT_MODEL = "all-MiniLM-L6-v2"   # meme modele que celui de Raphael
N_PER_BIN   = 8                    # nb d'exemples par tranche (diagnostic B)
OUT_DIR     = "task3_evaluation/outputs"

# Seuils du diagnostic D : un cas est "suspect" si la reponse est proche de
# la ground_truth (SBERT eleve = probablement correcte) mais que FactCC la
# classe comme hallucination (score bas).
SUSPECT_SBERT_MIN  = 0.55
SUSPECT_FACTCC_MAX = 0.15
N_SUSPECT          = 12            # nb de cas suspects imprimes

# --------------------------------------------------------------------------
# Chargement du modele SBERT (une seule fois)
# --------------------------------------------------------------------------
print(f"Loading SBERT model ({SBERT_MODEL}) ...")
model = SentenceTransformer(SBERT_MODEL)


def sbert_cosine(gens, gts):
    """Cosine similarity entre deux listes de strings, calculee en un batch.

    normalize_embeddings=True -> vecteurs L2-normalises, donc le produit
    scalaire ligne a ligne est exactement la cosine similarity.
    """
    gen_emb = model.encode(list(gens), normalize_embeddings=True,
                           show_progress_bar=False)
    gt_emb = model.encode(list(gts), normalize_embeddings=True,
                          show_progress_bar=False)
    return (gen_emb * gt_emb).sum(axis=1)


def word_count(s):
    """Nombre de mots d'une chaine (proxy simple de la verbosite)."""
    return len(str(s).split())


def short(s, n=300):
    """Tronque une chaine trop longue pour garder l'affichage lisible."""
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[:n] + " [...]"


def pearson(x, y):
    """Correlation de Pearson (lineaire) calculee avec numpy."""
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    """Correlation de Spearman = Pearson sur les rangs (robuste au non-lineaire)."""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------
# Boucle principale : un diagnostic par pipeline
# --------------------------------------------------------------------------
for pipeline, path in FACTCC_DIRS.items():
    print("\n" + "=" * 72)
    print(f"  {pipeline}   ({path})")
    print("=" * 72)

    # --- Chargement du dataset FactCC -> DataFrame pandas -----------------
    # Colonnes : source, gen, id, question, ground_truth, hop_type,
    #            pipeline, predictions, score   (score = score FactCC).
    df = load_from_disk(path).to_pandas()

    for col in ("gen", "ground_truth", "source"):
        df[col] = df[col].fillna("").astype(str)
    valid = df[(df["gen"].str.strip() != "")
               & (df["ground_truth"].str.strip() != "")].copy()
    print(f"Entrees valides (gen & ground_truth non vides) : "
          f"{len(valid)}/{len(df)}")

    # --- Calcul de la cosine SBERT pour chaque ligne ---------------------
    valid["sbert"] = sbert_cosine(valid["gen"], valid["ground_truth"])
    valid["gen_words"] = valid["gen"].apply(word_count)
    valid["gt_words"] = valid["ground_truth"].apply(word_count)

    # =====================================================================
    # DIAGNOSTIC A — longueurs gen vs ground_truth
    # =====================================================================
    print("\n--- A. Longueurs (en mots) ---")
    print(f"  gen          : moyenne={valid['gen_words'].mean():5.1f}  "
          f"mediane={valid['gen_words'].median():.0f}")
    print(f"  ground_truth : moyenne={valid['gt_words'].mean():5.1f}  "
          f"mediane={valid['gt_words'].median():.0f}")
    ratio = valid["gen_words"].mean() / max(valid["gt_words"].mean(), 1e-9)
    print(f"  ratio gen / ground_truth : {ratio:.1f}x")

    # =====================================================================
    # DIAGNOSTIC B — echantillons stratifies (SBERT bas / moyen / haut)
    # =====================================================================
    # Pour chaque exemple on imprime maintenant aussi le champ `source`
    # (la phrase de verite contre laquelle FactCC compare).
    vs = valid.sort_values("sbert").reset_index(drop=True)
    n = len(vs)
    bins = {
        "SBERT BAS  (reponses les plus eloignees)":  vs.iloc[:N_PER_BIN],
        "SBERT MOYEN":                               vs.iloc[n // 2 - N_PER_BIN // 2:
                                                              n // 2 + N_PER_BIN // 2],
        "SBERT HAUT (reponses les plus proches)":    vs.iloc[-N_PER_BIN:],
    }
    for label, chunk in bins.items():
        print(f"\n--- B. Echantillons : {label} ---")
        for _, r in chunk.iterrows():
            print(f"  [{r['id']} | {r['hop_type']}]  "
                  f"sbert={r['sbert']:.3f}  factcc={r['score']:.3f}")
            print(f"     gen          ({r['gen_words']:2d} mots) : {short(r['gen'])}")
            print(f"     ground_truth ({r['gt_words']:2d} mots) : {r['ground_truth']}")
            print(f"     source                : {short(r['source'])}")

    # =====================================================================
    # DIAGNOSTIC C — correlation FactCC <-> SBERT
    # =====================================================================
    pear = pearson(valid["score"].to_numpy(), valid["sbert"].to_numpy())
    spear = spearman(valid["score"].to_numpy(), valid["sbert"].to_numpy())
    print("\n--- C. Correlation FactCC <-> SBERT ---")
    print(f"  Pearson  r = {pear:+.3f}")
    print(f"  Spearman p = {spear:+.3f}")
    print("  seuil de significativite a n=547 : |r| > ~0.084")

    # =====================================================================
    # DIAGNOSTIC D — cas suspects : SBERT haut + FactCC bas, AVEC source
    # =====================================================================
    # Ce sont les cas ou la reponse est probablement correcte (proche de la
    # ground_truth) mais ou FactCC dit "hallucination". Le champ `source`
    # permet de trancher :
    #   - source = bonne phrase  -> FactCC se trompe sur ce domaine
    #   - source = phrase hors-sujet / vide -> bug de routage (build_source_index)
    suspect = valid[(valid["sbert"] >= SUSPECT_SBERT_MIN)
                    & (valid["score"] <= SUSPECT_FACTCC_MAX)]
    suspect = suspect.sort_values("sbert", ascending=False)

    # Pour information : le quadrant oppose (hedges qui passent FactCC).
    opposite = valid[(valid["sbert"] <= 0.15)
                     & (valid["score"] >= 0.85)]

    print(f"\n--- D. Cas suspects : SBERT >= {SUSPECT_SBERT_MIN} "
          f"ET FactCC <= {SUSPECT_FACTCC_MAX} ---")
    print(f"  {len(suspect)} cas (reponse probablement correcte mais "
          f"FactCC = hallucination)")
    print(f"  (pour info, quadrant oppose SBERT<=0.15 & FactCC>=0.85 : "
          f"{len(opposite)} cas)")
    print(f"  -> on imprime les {min(N_SUSPECT, len(suspect))} plus suspects "
          f"avec leur champ `source` :\n")
    for _, r in suspect.head(N_SUSPECT).iterrows():
        print(f"  [{r['id']} | {r['hop_type']}]  "
              f"sbert={r['sbert']:.3f}  factcc={r['score']:.3f}")
        print(f"     question     : {short(r['question'])}")
        print(f"     gen          : {short(r['gen'])}")
        print(f"     ground_truth : {r['ground_truth']}")
        print(f"     source       : {short(r['source'])}")
        print()

    # =====================================================================
    # Export CSV pour inspection manuelle dans un tableur
    # =====================================================================
    cols = ["id", "hop_type", "question", "gen", "ground_truth", "source",
            "gen_words", "gt_words", "score", "sbert"]
    out_path = f"{OUT_DIR}/diagnostic_{pipeline.lower()}.csv"
    valid[cols].rename(columns={"score": "factcc_score"}).to_csv(
        out_path, index=False)
    print(f"Export -> {out_path}")

print("\nTermine.")