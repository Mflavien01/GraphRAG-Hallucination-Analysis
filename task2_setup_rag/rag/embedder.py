from sentence_transformers import SentenceTransformer
import numpy as np


def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    """Embed chunks using SentenceTransformer. Add "embedding" key to each chunk dict with the embedding vector."""
    model = SentenceTransformer(model_name)
    #encode all chunk texts at once for efficiency
    texts=[]
    for chunk in chunks:
        texts.append(chunk["text"]) 
    embeddings = model.encode(texts, show_progress_bar=True)
    for chunk, embedding in zip(chunks, embeddings):  # link each chunk with its embedding
        chunk["embedding"] = embedding
    return chunks

# # Example usage:
# chunks=[
#     {
#   "id":       "ont_3_airport_test_1_chunk_0",  # id original + index chunk
#   "category": "airport",
#   "dataset":  "lettria",
#   "text":     "Abilene Regional Airport serves the city of Abilene, Texas, and is located approximately 5 miles southwest of the city center."
#     },
#     {
#     "id":       "ont_3_airport_test_2_chunk_0",
#     "category": "airport",
#     "dataset":  "lettria",
#     "text":     "Abilene Regional Airport serves the city of Abilene, Texas, and is located approximately 5 miles southwest of the city center."
# }
# ]
# embedded_chunks = embed_chunks(chunks)
# print(embedded_chunks)
        