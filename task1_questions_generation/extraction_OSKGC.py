import random
import xml.etree.ElementTree as ET

random.seed(42)

# Number of sentences to extract per dataset
dataset_info = {
    "1_Airport":              4,
    "1_Artist":               4,
    "1_Astronaut":            1,
    "1_Athlete":              4,
    "1_Building":             3,
    "1_CelestialBody":        2,
    "1_City":                 3,
    "1_ComicsCharacter":      2,
    "1_Company":              1,
    "1_Film":                 1,
    "1_Food":                 3,
    "1_MeanOfTransportation": 4,
    "1_Monument":             1,
    "1_MusicalWork":          1,
    "1_Politician":           4,
    "1_Scientist":            1,
    "1_SportsTeam":           3,
    "1_University":           1,
    "1_WrittenWork":          3,
    "2_Airport":              3,
    "2_Artist":               3,
    "2_Astronaut":            1,
    "2_Athlete":              3,
    "2_Building":             2,
    "2_CelestialBody":        2,
    "2_City":                 3,
    "2_ComicsCharacter":      1,
    "2_Company":              1,
    "2_Film":                 1,
    "2_Food":                 3,
    "2_MeanOfTransportation": 3,
    "2_Monument":             1,
    "2_MusicalWork":          1,
    "2_Politician":           3,
    "2_Scientist":            1,
    "2_SportsTeam":           2,
    "2_University":           1,
    "2_WrittenWork":          3,
    "3_Airport":              3,
    "3_Artist":               3,
    "3_Astronaut":            1,
    "3_Athlete":              3,
    "3_Building":             3,
    "3_CelestialBody":        2,
    "3_City":                 3,
    "3_ComicsCharacter":      1,
    "3_Company":              1,
    "3_Film":                 1,
    "3_Food":                 4,
    "3_MeanOfTransportation": 3,
    "3_Monument":             1,
    "3_MusicalWork":          1,
    "3_Politician":           3,
    "3_Scientist":            1,
    "3_SportsTeam":           2,
    "3_University":           1,
    "3_WrittenWork":          3,
}

with open("extracted_text_OSKGC.txt", "w", encoding="utf-8") as out:

    for dataset, n in dataset_info.items():

        # Parse the XML file and extract all <text> entries
        tree = ET.parse(f"./data/OSKGC/benchmark/data/dev/{dataset}.xml")
        root = tree.getroot()

        sentences = []
        for entry in root.findall(".//entry"):
            text = entry.find("text").text.strip()
            sentences.append(text)

        # Randomly pick n sentences and write them to the output file
        sampled = random.sample(sentences, n)
        for sentence in sampled:
            out.write(sentence + "\n")


