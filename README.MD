# Hallucination Analysis in RAG and GraphRAG Systems

## 📌 Project Overview
Large Language Models (LLMs) augmented with retrieval mechanisms (RAG) often generate hallucinations—statements not supported by the retrieved information . 
This semester project aims to study, in a concrete and empirical way, how the use of structured knowledge graphs in **GraphRAG** systems affects the frequency and nature of hallucinations compared to a classical vector-based **RAG** .

**Institution:** EURECOM
**Supervisors:** Carmelle Meli Songuon, Thibault Ehrhart, Pasquale Lisena, Raphael Troncy.

## 🎯 Objectives & Tasks
The project is structured around three main tasks:
1. **Task 1 - Dataset Expansion:** Generating a balanced set of simple and multi-hop questions using 50% text-based generation and 50% graph-based generation based on the *OSKGC* and *Text2KGBench-Lettria* datasets.
2. **Task 2 - Pipelines Setup:** Implementing two question-answering pipelines (Standard Vector RAG and GraphRAG) using LangChain.
3. **Task 3 - Evaluation:** Running controlled experiments to evaluate hallucinations quantitatively (using the Mirage framework ) and qualitatively (analyzing failure cases such as unsupported facts, incorrect relations, or invented entities ).
4. *(Bonus)* **Grounding Mechanisms:** Implementing graduated grounding mechanisms to explicitly control the degree of creativity during generation and study the trade-offs.


## 🛠️ Datasets
*   [Text2KGBench-LettrIA](https://drive.google.com/file/d/1GCEEHzHkTkaMbjgrKfYgIcZVE4-VixPe/view?usp=drive_link): A curated benchmark for ontology-guided Text-to-Knowledge-Graph systems containing 4,860 sentences and over 14,000 triples.
*   [OSKGC](https://github.com/HeraclesWang/OSKGC): A benchmark designed for knowledge graph construction based on a predefined ontology schema containing 10,183 text entries.
--------------------------------------------------------------------------------