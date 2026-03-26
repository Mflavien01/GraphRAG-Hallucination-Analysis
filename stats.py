import json
import os
from rdflib import Graph, RDF, OWL

total_sentences = 0
total_triplets = 0
stat_per_dataset = {}
set_of_properties = set()


for root, dirs, files in os.walk("data/Text2KGBench_LettrIA/"):
    for d in dirs:
        number_sentence=0
        number_triplet=0
        with open(f"data/Text2KGBench_LettrIA/{d}/ground_truth.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                number_sentence += 1
                number_triplet+= len(data["triples"])
                for triples in data["triples"]:
                    set_of_properties.add(triples['rel'])
            if d not in stat_per_dataset:
                stat_per_dataset[d] = {"sentences": number_sentence, "triplets": number_triplet}
            total_sentences += number_sentence
            total_triplets += number_triplet
        g=Graph()
        g.parse(f"data/Text2KGBench_LettrIA/{d}/ontology.ttl", format="turtle")
        for prop in g.subjects(RDF.type, OWL.ObjectProperty):
            set_of_properties.add(str(prop).split("#")[1])
        for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
            set_of_properties.add(str(prop).split("#")[1])
print(f"Total sentences: {total_sentences}")
print(f"Total triplets: {total_triplets}")
print(f"Statistics per dataset: {stat_per_dataset}")
print(f"Unique properties: {len(set_of_properties)}")