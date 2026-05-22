import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re


def build_bm25_index(chunks):
    """
    Construit un index BM25 depuis la liste de chunks.
    BM25 travaille sur des tokens (mots), pas sur des vecteurs.
    On tokenise simplement en minuscules et on split sur les espaces/ponctuation.
    """
    tokenized_corpus = []
    for chunk in chunks:
        # Tokenisation simple : minuscules + split sur non-alphanumérique
        tokens = re.findall(r'\w+', chunk["text"].lower())
        tokenized_corpus.append(tokens)
    
    return BM25Okapi(tokenized_corpus)

def search_hybrid(query, faiss_index, bm25_index, chunks, model, k=5):
    """
    Retrieval hybride : combine FAISS (sémantique) + BM25 (lexical)
    via Reciprocal Rank Fusion (RRF).

    RRF score pour un chunk i = 1/(rank_faiss(i) + 60) + 1/(rank_bm25(i) + 60)
    La constante 60 est standard dans la littérature (Cormack et al. 2009).
    Un chunk bien classé dans les DEUX retrieval obtient le meilleur score.
    """
    # ── FAISS : récupère top k*3 pour avoir assez d'overlap avec BM25 ──
    query_vector = model.encode(query).reshape(1, -1)
    distances, faiss_indices = faiss_index.search(query_vector, k * 3)

    # Construit un dict {idx_chunk: rank_faiss} (rank commence à 0)
    faiss_ranks = {}
    for rank, idx in enumerate(faiss_indices[0]):
        faiss_ranks[idx] = rank

    # ── BM25 : score tous les chunks, puis classe par score décroissant ──
    query_tokens = re.findall(r'\w+', query.lower())
    bm25_scores = bm25_index.get_scores(query_tokens)  # score pour chaque chunk

    # Classe les indices par score BM25 décroissant → donne le rank BM25
    bm25_ranked = sorted(range(len(bm25_scores)),
                         key=lambda i: bm25_scores[i], reverse=True)
    bm25_ranks = {idx: rank for rank, idx in enumerate(bm25_ranked)}

    # ── RRF : fusionne les deux rankings ─────────────────────────────────
    # On ne considère que les chunks présents dans au moins un des deux
    candidate_indices = set(faiss_ranks.keys()) | set(range(len(chunks)))

    rrf_scores = {}
    for idx in candidate_indices:
        rank_f = faiss_ranks.get(idx, k * 3 + 60)  # pénalité si absent du FAISS
        rank_b = bm25_ranks.get(idx, len(chunks))   # pénalité si absent du BM25
        rrf_scores[idx] = 1 / (rank_f + 60) + 1 / (rank_b + 60)

    # Top-k par score RRF décroissant
    top_k_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:k]

    # ── Retourne le même format que search() ─────────────────────────────
    results = []
    for idx in top_k_indices:
        results.append({
            "id":       chunks[idx]["id"],
            "category": chunks[idx]["category"],
            "dataset":  chunks[idx]["dataset"],
            "text":     chunks[idx]["text"],
            "distance": float(rrf_scores[idx])  # ici c'est le score RRF, pas L2
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
