import networkx as nx



def build_graph(triples):
    """Build a graph from a list of triples. Use of MultiDiGraph instead of DiGraph to allow multiple edges between the same nodes"""
    G = nx.MultiDiGraph()
    for t in triples:
        G.add_edge(t["subject"], t["object"], relation=t["predicate"], source=t["source"])
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

