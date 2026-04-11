import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

data_path = "data/OSKGC/benchmark/data/dev"
stats = []

for xml_file in sorted(os.listdir(data_path)):
    if not xml_file.endswith(".xml"):
        continue
    tree = ET.parse(os.path.join(data_path, xml_file))
    root = tree.getroot()
    sentences = 0
    triplets = 0
    for entry in root.findall(".//entry"):
        if entry.find("text") is not None:
            sentences += 1
        triplets += len(entry.findall(".//triples/triple"))
    stats.append({"dataset": xml_file.replace(".xml", ""), "sentences": sentences, "triplets": triplets})

df = pd.DataFrame(stats)
df["percentage"] = np.ceil((df["sentences"] / df["sentences"].sum() * 100).round(1))
df.loc[len(df)] = ["TOTAL", df["sentences"].sum(), df["triplets"].sum(), 100.0]
print(df.to_string(index=False))