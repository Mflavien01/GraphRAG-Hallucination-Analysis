import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def load_lettria(lettria_dir):
    """Load all entries from the Text2KGBench-LettrIA dataset.

    Returns a list of dicts with keys: id, category, sent, triples.
    """
    entries = []
    base = Path(lettria_dir)

    for category_dir in sorted(base.iterdir()):
        if not category_dir.is_dir():
            continue

        gt_file = category_dir / "ground_truth.jsonl"
        if not gt_file.exists():
            continue

        category = category_dir.name

        with open(gt_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                entries.append({
                    "id":       data.get("id", ""),
                    "category": category,
                    "sent":     data["sent"],
                    "triples":  [
                        {"sub": t["sub"], "rel": t["rel"], "obj": t["obj"]}
                        for t in data["triples"]
                    ]
                })

    print(f"[LettrIA] {len(entries)} entries loaded")
    return entries


def load_oskgc(oskgc_dir):
    """Load all entries from the OSKGC dataset.

    Returns a list of dicts with keys: id, category, sent, triples.
    """
    entries = []
    base = Path(oskgc_dir)

    for xml_file in sorted(base.glob("*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for entry in root.findall("entries/entry"):
            entry_id = entry.get("id", "")
            category = entry.get("category", "")

            text_node = entry.find("text")
            sent = text_node.text.strip() if text_node is not None else ""

            triples = []
            for triple in entry.findall(".//triple"):
                sub = triple.find("sub")
                rel = triple.find("rel")
                obj = triple.find("obj")
                if sub is not None and rel is not None and obj is not None:
                    triples.append({
                        "sub": sub.text.strip(),
                        "rel": rel.text.strip(),
                        "obj": obj.text.strip()
                    })

            if sent and triples:
                entries.append({
                    "id":       entry_id,
                    "category": category,
                    "sent":     sent,
                    "triples":  triples
                })

    print(f"[OSKGC] {len(entries)} entries loaded")
    return entries


def sample_proportional(entries, n, seed=42):
    """Sample n entries proportionally across categories.

    Each category contributes entries proportional to its share of the total.
    """
    random.seed(seed)

    by_category = defaultdict(list)
    for e in entries:
        by_category[e["category"]].append(e)

    total = len(entries)
    sampled = []

    for category, items in by_category.items():
        proportion = len(items) / total
        n_cat = max(1, round(proportion * n))
        picked = random.sample(items, min(n_cat, len(items)))
        sampled.extend(picked)

    random.shuffle(sampled)
    sampled = sampled[:n]

    print(f"  → {len(sampled)} entries selected out of {total}")
    return sampled
