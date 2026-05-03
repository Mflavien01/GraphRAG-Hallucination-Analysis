from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def build_statement_index(triples, model_name="all-MiniLM-L6-v2"):
    '''Build a FAISS index from the given triples. Each triple is converted to a statement string and embedded using a sentence transformer model. The embeddings are normalized for cosine similarity search.'''
    
    # Load the sentence transformer model
    model = SentenceTransformer(model_name)
    # (subject, predicate, object) → "subject -[predicate]-> object"
    statements = []
    for t in triples:
        statement = f"{t['subject']} -[{t['predicate']}]-> {t['object']}"
        statements.append(statement)
    
    #Embed all statements with L2 normalization. 
    embeddings = model.encode(statements, convert_to_numpy=True, normalize_embeddings=True) 
    embeddings = np.array(embeddings).astype("float32") # FAISS requires float32
    
    index = faiss.IndexFlatIP(embeddings.shape[1]) # Inner Product = cosine similarity on normalized vectors
    index.add(embeddings) #store embeddings in the FAISS index
    
    return index, statements, model


def retrieve_statements(question, index, statements, model, k=10):
    '''Given a question, retrieve the top-k most relevant statements from the index. The question is embedded using the same model and normalized for cosine similarity search.'''
    
    query_embedding = model.encode([question], normalize_embeddings=True) # embed and normalize the question
    query_embedding = np.array(query_embedding).astype("float32") # ensure correct dtype for FAISS

    scores, indices = index.search(query_embedding, k)  # search the index for the top-k most similar statement embeddings

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append(statements[idx]) 

    return results


# triples = [
#     {"subject": "Paris", "predicate": "capital_of", "object": "France"},
#     {"subject": "Berlin", "predicate": "capital_of", "object": "Germany"},
#     {"subject": "Madrid", "predicate": "capital_of", "object": "Spain"},
#     {"subject": "France", "predicate": "located_in", "object": "Europe"},
#     {"subject": "Eiffel Tower", "predicate": "located_in", "object": "Paris"},
# ]

# index, statements, model = build_statement_index(triples)

# print("\nStatements générés :")
# for s in statements:
#     print("-", s)

# question = "What is the capital of France?"

# results = retrieve_statements(question, index, statements, model, k=3)

# print("\nRésultats :")
# for r in results:
#     print("-", r)