from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_texts(texts, chunk_size=500, chunk_overlap=50):
    """Chunk texts into smaller pieces of max chunk_size with chunk_overlap between chunks. Return list of dicts with same keys as input + "chunk_id" and "chunk_text"."""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for text in texts:
        chunk=splitter.split_text(text["text"])
        for chunk_id, chunk_text in enumerate(chunk):
            all_chunks.append(
                    {
                        "id": text["id"] + "_chunk_" + str(chunk_id),
                        "category": text["category"],
                        "dataset": text["dataset"],
                        "text": chunk_text,
                }
            )

    return all_chunks

# # Example usage:
# texts =[
#   {"id": "ont_3_airport_test_1", "category": "airport", "dataset": "OSKGC", "text": "Angola International Airport is located at Ícolo e Bengo."},
#   {"id": "ont_3_airport_test_2", "category": "airport", "dataset": "OSKGC", "text": "Luanda International Airport is located at Luanda."},
# ]
# chunks = chunk_texts(texts)
# print(chunks)