import json
import os
from rdflib import Graph, RDF, OWL
import pandas as pd

total_sentences = 0
total_triplets = 0
stat_per_dataset = {}
set_of_properties = set()


for root, dirs, files in os.walk("data/Text2KGBench_LettrIA/"):
    for d in dirs:
        number_sentence = 0
        number_triplet = 0
        dataset_properties = set()
        with open(f"data/Text2KGBench_LettrIA/{d}/ground_truth.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                number_sentence += 1
                number_triplet += len(data["triples"])
                for triples in data["triples"]:
                    set_of_properties.add(triples['rel'])
                    dataset_properties.add(triples['rel'])
            if d not in stat_per_dataset:
                stat_per_dataset[d] = {"sentences": number_sentence, "triplets": number_triplet, "properties": sorted(dataset_properties)}
            total_sentences += number_sentence
            total_triplets += number_triplet
        g = Graph()
        g.parse(f"data/Text2KGBench_LettrIA/{d}/ontology.ttl", format="turtle")
        for prop in g.subjects(RDF.type, OWL.ObjectProperty):
            set_of_properties.add(str(prop).split("#")[1])
        for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
            set_of_properties.add(str(prop).split("#")[1])

rows = [{"dataset": d, "sentences": v["sentences"], "triplets": v["triplets"], "properties": v["properties"]} for d, v in stat_per_dataset.items()]
rows.append({"dataset": "TOTAL", "sentences": total_sentences, "triplets": total_triplets, "properties": sorted(set_of_properties)})

df_stats = pd.DataFrame(rows)
df_properties = pd.DataFrame(sorted(set_of_properties), columns=["property"])

with pd.ExcelWriter("stats.xlsx", engine="openpyxl") as writer:
    df_stats.to_excel(writer, sheet_name="Stats per dataset", index=False)
    df_properties.to_excel(writer, sheet_name="Unique properties", index=False)
