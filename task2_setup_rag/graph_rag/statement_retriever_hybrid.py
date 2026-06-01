"""
GraphRAG statement retriever — version hybride FAISS (cosine) + BM25 + cross-encoder

Même principe que rag/vector_store.py search_hybrid(), appliqué aux statements KG
au lieu des chunks texte.

Différence vs statement_retriever.py :
- build_statement_index_hybrid() construit deux index : FAISS + BM25
- retrieve_statements_hybrid() constitue un pool FAISS∪BM25 puis rerank cross-encoder
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
import re
from rank_bm25 import BM25Okapi

# Cross-encoder loaded once at module level (shared across all calls)
_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


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
    """Hybrid retrieval over KG statements: FAISS cosine + BM25 pool → cross-encoder rerank.

    Stage 1 — recall : union des top k*6 candidats FAISS et top k*6 BM25.
    Stage 2 — rerank : cross-encoder score chaque paire (question, statement)
               et retourne les k meilleurs.
    """
    candidate_k = k * 6

    # FAISS: top candidate_k semantic candidates
    query_emb = model.encode([question], normalize_embeddings=True)
    query_emb = np.array(query_emb).astype("float32")
    _, faiss_indices = faiss_index.search(query_emb, candidate_k)
    faiss_set = {int(idx) for idx in faiss_indices[0] if idx >= 0}

    # BM25: top candidate_k lexical candidates
    query_tokens = re.findall(r'\w+', question.lower())
    bm25_scores = bm25_index.get_scores(query_tokens)
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:candidate_k]
    bm25_set = set(bm25_top)

    # Pool: union of both sets
    candidate_indices = list(faiss_set | bm25_set)

    # Cross-encoder reranking
    candidate_texts = [statements[idx] for idx in candidate_indices]
    pairs = [(question, text) for text in candidate_texts]
    ce_scores = _cross_encoder.predict(pairs)

    ranked = sorted(zip(candidate_indices, ce_scores), key=lambda x: x[1], reverse=True)[:k]
    return [statements[idx] for idx, _ in ranked]
