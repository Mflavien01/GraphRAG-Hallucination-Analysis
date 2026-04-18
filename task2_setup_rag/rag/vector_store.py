import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


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
