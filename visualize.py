import json
from pyvis.network import Network

def visualize_graph(graph_data, output_file):
    # print(graph_data[0])
    net = Network(directed=True)
    for i in range(len(graph_data)):
        for triplet in graph_data[i]['triples']:
            sujet = triplet['sub'].lower()
            relation = triplet['rel'].lower()
            objet = triplet['obj'].lower()
            net.add_node(sujet, label=sujet)
            net.add_node(objet, label=objet)
            net.add_edge(sujet, objet, title=relation)
    net.save_graph(output_file)
        
        
        
file="data/Text2KGBench_LettrIA/airport/ground_truth.jsonl"
with open(file, 'r') as f:
    graph_data = [json.loads(line) for line in f]
visualize_graph(graph_data, 'graph2.html')

