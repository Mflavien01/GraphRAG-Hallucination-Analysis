import sys
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
sys.path.append(os.path.dirname(__file__))

from chunker import chunk_texts
from embedder import embed_chunks
from loader import load_all_texts
from vector_store import build_index, search


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(Path(__file__).parent.parent / "llm")) # add llm/ to path for llm_interface
from llm_interface import *

_texts = load_all_texts()
_chunks = chunk_texts(_texts)
_model = SentenceTransformer("all-MiniLM-L6-v2")
_embedded_chunks = embed_chunks(_chunks, model_name="all-MiniLM-L6-v2")
_index = build_index(_embedded_chunks)


def run_rag(question, llm: BaseLLM):
    results=search(question, _index, _embedded_chunks, _model) # search for the most relevant chunks in the index
    context_texts = "\n\n".join([result["text"] for result in results]) # concatenate the retrieved texts to provide as context to the LLM
    prompt = f"""Given the following retrieved context:
{context_texts}
Answer the following question based only on the context above:
{question}"""


    llm_answer = llm.generate(prompt)
    return {
        "question": question,
        "context": results,
        "context_texts": context_texts,
        "llm_answer": llm_answer
    }

    

