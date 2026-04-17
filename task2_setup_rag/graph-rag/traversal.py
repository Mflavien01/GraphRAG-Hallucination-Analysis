from collections import deque
import networkx as nx


def traverse(graph, anchor_nodes, k=2, max_triples=50):
    queue = deque([(node, 0) for node in anchor_nodes]) 
    visited = set(anchor_nodes)
    results = set()
    triple_count = 0
    while queue:
        current, depth = queue.popleft()
        if depth >=k:
            continue
        for source, target, key, data in graph.edges(current, keys=True,data=True):
            results.add(f"{source} -[{data['relation']}]-> {target}")
            triple_count += 1
            if target not in visited:
                visited.add(target)
                queue.append((target, depth + 1))
            if triple_count >= max_triples:
                return list(results)
    return list(results)


# G = nx.MultiDiGraph()

# # Nodes + edges (mini dataset clair)
# G.add_edge("Angola", "Luanda", relation="capital")
# G.add_edge("Luanda", "Africa", relation="located_in")
# G.add_edge("Angola", "Ícolo e Bengo", relation="has_region")
# G.add_edge("Ícolo e Bengo", "Luanda", relation="near")
# G.add_edge("France", "Paris", relation="capital")
# G.add_edge("Paris", "Europe", relation="located_in")
# anchors = ["Angola"]
# result = traverse(G, anchors, k=2)
# print(result)
# '''
# [
#  "Angola -[capital]-> Luanda",
#  "Angola -[has_region]-> Ícolo e Bengo",
#  "Luanda -[located_in]-> Africa",
#  "Ícolo e Bengo -[near]-> Luanda"
# ]
# '''

# anchors = ["France"]
# result = traverse(G, anchors, k=2)
# print(result)
# '''[
#  "France -[capital]-> Paris",
#  "Paris -[located_in]-> Europe"
# ]'''
