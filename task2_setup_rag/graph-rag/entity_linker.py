from rapidfuzz import process
import networkx as nx


def link_entities(question, graph, threshold):
    words = question.lower().split()
    ngrams = []
    
    # generate n-grams from the question n-grams of size 1, 2, and 3
    for n in range(1, 4):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            ngrams.append(ngram)
    
    # define a set to avoid duplicate matches
    matches = set()
    # get the list of nodes in the graph
    nodes = list(graph.nodes())
    
    # use rapidfuzz to find the best matching node for each n-gram
    for ngram in ngrams:
        result = process.extractOne(ngram, nodes, score_cutoff=threshold, processor=str.lower) #compare the ngram to all the nodes and return a list sorted by similarity score
        if result:
            match, score, _ = result
            matches.add(match)
    
    return list(matches)

# question = "Who is the president of France"
# G = nx.Graph()
# G.add_node("Emmanuel Macron")
# G.add_node("France")
# G.add_edge("Emmanuel Macron", "France", relation="president_of")
# print(G.nodes())
# linked_entities = link_entities(question, G, threshold=80)
# print(linked_entities)