from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_texts_parent_child(
    texts,
    parent_chunk_size=1500,
    child_chunk_size=150,
    chunk_overlap=20,
):
    """
    Parent/child chunking strategy:
      - Parent chunks (large, ~parent_chunk_size tokens): given to the LLM as context.
      - Child chunks (small, ~child_chunk_size tokens): indexed in FAISS for precise retrieval.

    Each child chunk has a 'parent_id' field pointing to its parent chunk id.
    Returns (parent_chunks, child_chunks).
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=chunk_overlap
    )

    parent_chunks = []
    child_chunks = []

    for text in texts:
        parents = parent_splitter.split_text(text["text"])
        for p_idx, parent_text in enumerate(parents):
            pid = f"{text['id']}_parent_{p_idx}"
            parent_chunks.append(
                {
                    "id": pid,
                    "category": text["category"],
                    "dataset": text["dataset"],
                    "text": parent_text,
                }
            )
            children = child_splitter.split_text(parent_text)
            for c_idx, child_text in enumerate(children):
                child_chunks.append(
                    {
                        "id": f"{pid}_child_{c_idx}",
                        "parent_id": pid,
                        "category": text["category"],
                        "dataset": text["dataset"],
                        "text": child_text,
                    }
                )

    return parent_chunks, child_chunks
