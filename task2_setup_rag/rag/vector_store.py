import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re

# Cross-encoder loaded once at module level (shared across all calls)
_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def build_bm25_index(chunks):
    """
    Construit un index BM25 depuis la liste de chunks.
    BM25 travaille sur des tokens (mots), pas sur des vecteurs.
    On tokenise simplement en minuscules et on split sur les espaces/ponctuation.
    """
    tokenized_corpus = []
    for chunk in chunks:
        tokens = re.findall(r'\w+', chunk["text"].lower())
        tokenized_corpus.append(tokens)

    return BM25Okapi(tokenized_corpus)

def search_hybrid(query, faiss_index, bm25_index, chunks, model, k=5):
    """
    Retrieval hybride : combine FAISS (sémantique) + BM25 (lexical) pour
    constituer un pool de candidats, puis reranking par cross-encoder.

    Stage 1 — recall : union des top k*6 candidats FAISS et top k*6 BM25.
    Stage 2 — rerank : cross-encoder score chaque paire (query, chunk)
               et retourne les k meilleurs.
    """
    candidate_k = k * 6

    # ── FAISS : top candidate_k candidats sémantiques ──────────────────
    query_vector = model.encode(query).reshape(1, -1)
    distances, faiss_indices = faiss_index.search(query_vector, candidate_k)
    faiss_set = {int(idx) for idx in faiss_indices[0] if idx >= 0}

    # ── BM25 : top candidate_k candidats lexicaux ───────────────────────
    query_tokens = re.findall(r'\w+', query.lower())
    bm25_scores = bm25_index.get_scores(query_tokens)
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:candidate_k]
    bm25_set = set(bm25_top)

    # ── Pool : union des deux ensembles ─────────────────────────────────
    candidate_indices = list(faiss_set | bm25_set)

    # ── Cross-encoder reranking ──────────────────────────────────────────
    candidate_texts = [chunks[idx]["text"] for idx in candidate_indices]
    pairs = [(query, text) for text in candidate_texts]
    ce_scores = _cross_encoder.predict(pairs)

    ranked = sorted(zip(candidate_indices, ce_scores), key=lambda x: x[1], reverse=True)[:k]

    # ── Retourne le même format que search() ────────────────────────────
    results = []
    for idx, score in ranked:
        results.append({
            "id":       chunks[idx]["id"],
            "category": chunks[idx]["category"],
            "dataset":  chunks[idx]["dataset"],
            "text":     chunks[idx]["text"],
            "distance": float(score)  # cross-encoder score (higher = more relevant)
        })
    return results

def build_index(chunks):
    matrix = np.array([chunk["embedding"] for chunk in chunks]) #get all embeddings from all chunks and put them in a matrix
    index = faiss.IndexFlatL2(matrix.shape[1])  # index in FAISS and use L2 distance (euclidean) for similarity search
    index.add(matrix) #put the matrix in the index
    return index
    

def search(query, index, chunks, model, k=5):
    results = []
    query_vector = model.encode(query).reshape(1, -1) # reshape for faiss for exactly one query but FAISS can also handle batches of queries if needed
    distances, indices = index.search(query_vector, k) # search in the index and get the indices of the k closest chunks
    for rank, idx in enumerate(indices[0]): # FAISS return a ranking of the closest chunks. Rank is the position in the ranking (0 is the closest), idx is the index of the chunk in the original list of chunks
        results.append(
            {
                "id": chunks[idx]["id"],
                "category": chunks[idx]["category"],
                "dataset": chunks[idx]["dataset"],
                "text": chunks[idx]["text"],
                "distance": distances[0][rank]
            }
        )
    return results






# chunks = [
#     {"id": 1, "category": "sport", "dataset": "lettria", "text": "Le football est un sport de ballon très populaire dans le monde entier."},
#     {"id": 2, "category": "sport", "dataset": "lettria", "text": "Le basketball est un sport de ballon qui se joue avec les mains et qui est très populaire aux États-Unis."},
#     {"id": 3, "category": "nourriture", "dataset": "lettria", "text": "La pizza est un plat italien à base de pâte, de sauce tomate et de fromage, souvent garnie de divers ingrédients."},
#     {"id": 4, "category": "nourriture", "dataset": "lettria", "text": "La salade est un plat composé de légumes frais, souvent accompagné d'une vinaigrette."}
# ]


# # -------------------------
# # 2. Modèle
# # -------------------------
# model = SentenceTransformer("all-MiniLM-L6-v2")


# # -------------------------
# # 3. Embedding des chunks
# # -------------------------
# def embed_chunks(chunks, model):
#     texts = []

#     # extraire les textes
#     for chunk in chunks:
#         texts.append(chunk["text"])

#     # encoder
#     embeddings = model.encode(texts)

#     # rattacher embeddings aux chunks
#     for i in range(len(chunks)):
#         chunks[i]["embedding"] = embeddings[i]

#     return chunks


# # -------------------------
# # 6. Pipeline complet
# # -------------------------
# chunks = embed_chunks(chunks, model)
# index = build_index(chunks)

# # test
# print("Search results for 'sport de ballon':")
# search("sport de ballon", index, chunks, model)
# print("\nSearch results for 'nourriture saine':")
# search("nourriture saine", index, chunks, model)
