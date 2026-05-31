"""
GraphRAG statement retriever — version hybride FAISS (cosine) + BM25 (RRF)

Même principe que rag/vector_store.py search_hybrid(), appliqué aux statements KG
au lieu des chunks texte.

Différence vs statement_retriever.py :
- build_statement_index_hybrid() construit deux index : FAISS + BM25
- retrieve_statements_hybrid() fusionne les deux rankings via RRF
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re
from rank_bm25 import BM25Okapi


def build_statement_index_hybrid(triples, model_name="all-MiniLM-L6-v2"):
    """Build FAISS (cosine) + BM25 indices from KG triples.

    Returns (faiss_index, bm25_index, statements, model).
    """
    model = SentenceTransformer(model_name)

    statements = [
        f"{t['subject']} -[{t['predicate']}]-> {t['object']}"
        for t in triples
    ]

    # FAISS index — cosine similarity via normalized inner product (same as base GraphRAG)
    embeddings = model.encode(statements, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)

    # BM25 index — tokenize statement strings (subject, predicate, object words)
    tokenized = [re.findall(r'\w+', s.lower()) for s in statements]
    bm25_index = BM25Okapi(tokenized)

    return faiss_index, bm25_index, statements, model


def retrieve_statements_hybrid(question, faiss_index, bm25_index, statements, model, k=10):
    """Hybrid retrieval over KG statements: FAISS cosine + BM25 via RRF.

    RRF score for statement i = 1/(rank_faiss(i) + 60) + 1/(rank_bm25(i) + 60)
    Constant 60 follows Cormack et al. (2009).
    """
    # FAISS: retrieve top k*3 candidates for overlap with BM25
    query_emb = model.encode([question], normalize_embeddings=True)
    query_emb = np.array(query_emb).astype("float32")
    _, faiss_indices = faiss_index.search(query_emb, k * 3)
    faiss_ranks = {int(idx): rank for rank, idx in enumerate(faiss_indices[0])}

    # BM25: score all statements, then rank by descending score
    query_tokens = re.findall(r'\w+', question.lower())
    bm25_scores = bm25_index.get_scores(query_tokens)
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_ranks = {idx: rank for rank, idx in enumerate(bm25_ranked)}

    # RRF: consider all statements (present in at least one ranking)
    candidate_indices = set(faiss_ranks.keys()) | set(range(len(statements)))
    rrf_scores = {}
    for idx in candidate_indices:
        rank_f = faiss_ranks.get(idx, k * 3 + 60)   # penalty if absent from FAISS top
        rank_b = bm25_ranks.get(idx, len(statements)) # penalty if absent from BM25
        rrf_scores[idx] = 1 / (rank_f + 60) + 1 / (rank_b + 60)

    top_k = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:k]
    return [statements[i] for i in top_k]
