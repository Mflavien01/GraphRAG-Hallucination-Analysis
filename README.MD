# Hallucination Analysis in RAG and GraphRAG Systems

> *How does the retrieval mechanism — not the generator — shape **when**, **why**, and **how** a large language model hallucinates?*

An empirical study comparing **classical RAG** (dense retrieval over text chunks)
with **GraphRAG** (retrieval over ontology-typed Knowledge-Graph triples), holding
the generator fixed and varying *only* the retrieval stage. Six reproducible
pipelines are evaluated on **547 validated question–answer pairs** spanning
single- and multi-hop reasoning over two ISWC 2025 benchmarks, scored with the
**MIRAGE / FactCC** framework and a Sentence-BERT correctness cross-check.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue.svg">
  <img alt="FAISS" src="https://img.shields.io/badge/retrieval-FAISS%20%2B%20BM25-teal.svg">
  <img alt="Evaluation" src="https://img.shields.io/badge/eval-MIRAGE%20%2F%20FactCC-orange.svg">
  <img alt="Status" src="https://img.shields.io/badge/status-complete-success.svg">
</p>

📄 **Full report:** [`graphrag-hallucination-report.pdf`](/docs/graphrag-hallucination-report.pdf) · **Authors:** Flavien Mathieu, Raphael Partouche (EURECOM)
**Supervisors:** Raphael Troncy, Pasquale Lisena, Carmelle Meli Songuon, Thibault Ehrhart

---

## TL;DR — Key findings

- **Hybrid recall + cross-encoder reranking on the *text* pipeline (RAG+CE) gives the highest mean faithfulness and answer correctness**, improving over plain RAG on *every* axis. A paired significance test, however, shows the gain over RAG is within confidence intervals; the statistically robust gap is between the text pipelines and GraphRAG, and it holds even on graph-derived questions.
- **Statistical significance:** RAG+CE vs RAG is not significant (paired bootstrap + permutation, p ≈ 0.40), whereas RAG vs GraphRAG is (p < 0.01), even on graph-derived questions (p ≈ 0.04).
- **GraphRAG trails** through three distinct, separable failure modes: leaf-node abstention, multi-hop context gaps, and surface-form mismatch with the text-based scorer.
- **A novel dual *Graph-Chunk* pipeline** — fusing reranked text chunks and KG triples through `[CHUNK]`/`[KG]`-labelled context — attains the **lowest abstention rate of all pipelines (10.4%)** and recovers most of GraphRAG's multi-hop deficit.
- **Retrieval depth is paradigm-dependent:** text pipelines saturate at small `k≈4`, graph pipelines need `k≈12–20`.
- **Methodology contribution:** we surface and correct a score-orientation inversion in the MIRAGE/FactCC interface (`faithfulness = 1 − score`), validated by the sign-flip of its Pearson correlation with SBERT (≈ −0.24 → ≈ +0.24).

### Results at a glance

FactCC faithfulness (answered entries only, coverage errors excluded), abstention
rate, hallucination rate, and SBERT answer-correctness across all six pipelines on
the full 547-question test set:

| Pipeline | `k` | Answered | Coverage err. ↓ | Faithfulness ↑ | Halluc. ↓ | SBERT ↑ |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| RAG baseline      | 4      | 453/547 | 17.2% | 0.547 | 44.6% | 0.446 |
| GraphRAG baseline | 12     | 364/547 | 33.5% | 0.502 | 49.2% | 0.410 |
| **RAG + CE**      | 4      | 483/547 | 11.7% | **0.560** | **43.9%** | **0.464** |
| GraphRAG + CE     | 13     | 397/547 | 27.4% | 0.499 | 49.4% | 0.437 |
| Graph-Chunk + CE  | 20+20  | 490/547 | **10.4%** | 0.527 | 47.1% | 0.454 |
| RAG Parent-Child  | 4      | 464/547 | 15.2% | 0.534 | 46.1% | 0.445 |

*Best per column in **bold**. `k` is the retrieval depth frozen from the K-sweep;
Graph-Chunk uses a fixed per-leg budget (20 chunks + 20 triples).*

---

## Research question & contributions

The central question: *how do the retrieval mechanisms of RAG and GraphRAG
influence the frequency, nature, and typology of hallucinations produced by an
LLM?* The goal is **not to eliminate hallucinations** but to characterise them.

This repository contributes:

1. **A balanced 547-pair evaluation dataset** over two KG benchmarks
   (Text2KGBench-LettrIA + OSKGC), built by proportional stratified sampling and
   unfolded into T5-derived (200), single-hop (199), and multi-hop (148)
   questions with dual text/graph provenance.
2. **Six reproducible QA pipelines** sharing one generator (see below), isolating
   retrieval as the only variable.
3. **A novel dual Graph-Chunk pipeline** that lets the *generator* fuse text and
   graph evidence at generation time rather than at the retrieval stage.
4. **A combined hallucination evaluation** (MIRAGE/FactCC + SBERT) with a
   four-category qualitative failure taxonomy.
5. **An evaluation-methodology analysis**: the FactCC score-orientation fix, a
   demonstration of FactCC's surface-form bias against ontology-derived answers,
   and a systematic retrieval-depth (`k`) study.

---

## The six pipelines

All pipelines share the **same generator** — `Qwen3-30B-A3B-Instruct-2507` (a
30B/3B-active Mixture-of-Experts model) served through the EURECOM self-hosted
LiteLLM gateway — and the **same source corpora**, so any difference in
hallucination behaviour is attributable to retrieval alone.

| Pipeline | Retriever | Reranker | Idea |
|---|---|---|---|
| **RAG** | FAISS `IndexFlatL2` over text chunks | — | Dense semantic retrieval baseline |
| **GraphRAG** | FAISS `IndexFlatIP` over verbalised `A-[rel]->B` triples | — | Ontology-typed structured retrieval |
| **RAG + CE** | FAISS ∪ BM25 (chunks) | Cross-Encoder | Hybrid recall, lexical + semantic, then rerank |
| **GraphRAG + CE** | FAISS ∪ BM25 (triples) | Cross-Encoder | Same recall-and-rerank on the triple side |
| **Graph-Chunk + CE** | *two independent* hybrid retrievers | Cross-Encoder | Concatenate `[CHUNK]` + `[KG]` context; LLM fuses |
| **RAG Parent-Child** | FAISS over child chunks → parent context | — | Retrieve fine, generate on wide context (ablation) |

**Shared components (frozen across pipelines):**

- **Encoder** — `all-MiniLM-L6-v2` (sentence-transformers) for every dense index.
- **Sparse retriever** — BM25; the hybrid pool is the **union of the top-`6k` FAISS and BM25 candidates**.
- **Cross-encoder** — `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Generator** — `Qwen3-30B-A3B-Instruct-2507` (MoE, 3B active) via the EURECOM LiteLLM gateway.

The cross-encoder **superseded an earlier Reciprocal Rank Fusion (RRF) variant**,
reported in the paper as a measured intermediate iteration (Graph-Chunk:
0.533 @ 13.5% with RRF → 0.540 @ 11.0% with the cross-encoder). Each pipeline
writes one `*_results.jsonl` (`rag`, `graphrag`, `rag_hybrid`,
`graphrag_hybrid`, `graph_chunk`, `rag_parent_child`), 547 rows each.

---

## Repository structure

```
.
├── raport.tex                     # Full research report (LaTeX)
├── data/                          # Datasets (gitignored — see Setup)
├── stats/                         # Category-level dataset distributions
│   ├── stats_Text2KG.py
│   └── stats_OSKGC.py
├── task1_questions_generation/    # Task 1 — T5 (text) + LLM (graph) question generation
│   └── output/                    #   questions_{t5,llm}_{lettria,oskgc}.jsonl
├── task2_setup_rag/               # Task 2 — the six retrieval pipelines
│   ├── rag/                       #   chunking, embedding, FAISS, hybrid + parent-child
│   ├── graph_rag/                 #   statement retriever, hybrid, graph-chunk
│   ├── llm/                       #   EurecomLLM gateway wrapper
│   ├── run_pipelines.py           #   orchestrator (checkpoint/resume + retry)
│   ├── run_k_sweep.py             #   retrieval-depth sweep
│   └── output/                    #   *_results.jsonl (547 rows each)
├── task3_evaluation/              # Task 3 — MIRAGE / FactCC evaluation
│   ├── prepare_mirage_input.py    #   build MIRAGE-ready CSVs (+ abstention detector)
│   ├── run_mirage.py              #   run FactCC per pipeline
│   ├── analyze_results.py         #   global / per-hop / SBERT analysis + CSV export
│   ├── diagnostic_{factcc,sbert}.py
│   ├── mirage_lib/                #   vendored MIRAGE library
│   ├── inputs/  outputs/          #   CSVs + HuggingFace Arrow score datasets
│   └── k_sweep_eval.py
├── requirements.txt
└── .env                           # Local paths and API keys (not committed)
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/Mflavien01/GraphRAG-Hallucination-Analysis.git
cd GraphRAG-Hallucination-Analysis
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Main dependencies: `transformers`, `torch`, `sentence-transformers`, `faiss-cpu`,
`rank-bm25`, `datasets` (HuggingFace, used by MIRAGE), `openai` (Task 1 graph QG),
`rdflib`, `networkx`, `rapidfuzz`, `json-repair`, `pandas`, `numpy`,
`python-dotenv`, `langchain-*`.

### 2. Download the datasets

The `data/` folder is gitignored; both datasets are downloaded manually.

**Text2KGBench-LettrIA** — download from the authors' Google Drive
([link](https://drive.google.com/file/d/1GCEEHzHkTkaMbjgrKfYgIcZVE4-VixPe/view?usp=drive_link)),
unzip under `data/` so that the layout is:

```
data/Text2KGBench_LettrIA/
├── airport/ground_truth.jsonl
├── artist/ground_truth.jsonl
├── ...
└── writtenwork/ground_truth.jsonl
```

Each `ground_truth.jsonl` pairs a source sentence with its typed triples.

**OSKGC** — clone the official repository:

```bash
git clone https://github.com/HeraclesWang/OSKGC.git data/OSKGC
```

The XML entries used live at `data/OSKGC/benchmark/data/dev/*.xml`.

### 3. Configure environment variables

Create a `.env` at the repository root:

```bash
LETTRIA_DIR=data/Text2KGBench_LettrIA
OSKGC_DIR=data/OSKGC/benchmark/data/dev

# OpenAI API — Task 1 graph-based question generation (gpt-4o)
OPENAI_API_KEY=your_openai_key_here

# EURECOM LiteLLM gateway — Task 2 generator (Qwen3-30B)
EURECOM_LLM_URL=https://...
EURECOM_LLM_KEY=your_eurecom_key_here
EURECOM_LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
```

---

## Reproducing the study

### Dataset statistics (drives the proportional sampler)

```bash
python stats/stats_Text2KG.py
python stats/stats_OSKGC.py
```

### Task 1 — Question generation

```bash
python task1_questions_generation/generate_t5.py    # text-derived, fine-tuned T5
python task1_questions_generation/generate_llm.py    # graph-derived, strictly-prompted LLM (gpt-4o)
```

Produces `questions_{t5,llm}_{lettria,oskgc}.jsonl` (100 entries each) in
`task1_questions_generation/output/`. Each line carries the originating sentence,
its triples, and the generated question(s) with explicit single-/multi-hop labels.

### Task 2 — Run the pipelines

```bash
# Everything (checkpointed — re-run to resume after a gateway outage):
python task2_setup_rag/run_pipelines.py all

# Or a single pipeline:
python task2_setup_rag/run_pipelines.py {rag|graphrag|hybrid|graphrag-hybrid|graph-chunk|parent-child}
```

Outputs land in `task2_setup_rag/output/*_results.jsonl` (547 rows each). The
orchestrator skips already-completed `(id, hop_type)` pairs and retries transient
API failures with exponential backoff (up to 300 s). Retrieval depths can be
explored with `run_k_sweep.py`.

### Task 3 — MIRAGE / FactCC evaluation

```bash
# 1. Build MIRAGE-ready CSVs from the Task 2 outputs
python task3_evaluation/prepare_mirage_input.py

# 2. Run FactCC per pipeline (writes HuggingFace Arrow datasets)
python task3_evaluation/run_mirage.py --pipeline both             # rag + graphrag
python task3_evaluation/run_mirage.py --pipeline hybrid           # rag_hybrid
python task3_evaluation/run_mirage.py --pipeline graphrag-hybrid
python task3_evaluation/run_mirage.py --pipeline graph-chunk
python task3_evaluation/run_mirage.py --pipeline parent-child

# 3. Analyse: global / per-hop_type / numeric-vs-text / SBERT / sanity correlation
python task3_evaluation/analyze_results.py
```

Score datasets land in `task3_evaluation/outputs/factcc_<pipeline>/`; per-row
scores are exported to `factcc_<pipeline>_scores.csv`.

**Note on score orientation.** MIRAGE exports `score = P(incorrect)`. All analysis
uses `faithfulness = 1 − score` so that *higher = better*. The standalone
`diagnostic_factcc.py` / `diagnostic_sbert.py` scripts document this orientation
fix and the FactAcc (REBEL) unsuitability check.

---

## Evaluation methodology notes

- **Coverage errors vs. hallucinations.** Explicit abstentions (*"the context does
  not contain information about X"*) are detected by a regex-based
  `is_coverage_error` matcher on the answer text — independent of FactCC — and
  excluded from faithfulness, then reported separately as a coverage-error rate.
  This prevents GraphRAG's leaf-node abstentions from inflating its apparent
  hallucination rate.
- **SBERT as a source-independent cross-check.** Cosine similarity between the
  generated answer and the ground-truth reference, immune to FactCC's surface-form
  confounds. Its positive correlation with corrected faithfulness validates the
  score orientation.
- **Why FactCC, not FactAcc.** REBEL-based FactAcc needs full-sentence context to
  extract triples; on 1–3 word KG answers it degenerates (mean ≈ 0.055, mostly
  zeros), so FactCC is used throughout.

---

## Results by question type (FactCC faithfulness)

| Pipeline | T5 | Single-hop | Multi-hop |
|---|:--:|:--:|:--:|
| RAG               | 0.636 | 0.524 | 0.414 |
| GraphRAG          | 0.610 | 0.493 | 0.314 |
| **RAG + CE**      | **0.663** | **0.525** | **0.438** |
| GraphRAG + CE     | 0.552 | 0.516 | 0.342 |
| Graph-Chunk + CE  | 0.614 | 0.499 | 0.426 |

Multi-hop is the divergence point: plain GraphRAG collapses (0.314, ~10 points
below RAG), and both hybrids recover it. The optimal retrieval depth differs by
paradigm (text `k*≈4`, graph `k*≈12–20`), and the ranking is stable across `k`.

---

## References

- **Text2KGBench-LettrIA** — Plu et al. *A Refined Benchmark for Text2Graph Systems.* ISWC 2025. [paper](https://ceur-ws.org/Vol-4041/paper3.pdf)
- **OSKGC** — Wang & Iwaihara. *A Benchmark for Ontology Schema-based KG Construction from Text.* ISWC 2025. [paper](https://ceur-ws.org/Vol-4041/paper1.pdf) · [repo](https://github.com/HeraclesWang/OSKGC)
- **T5 QG model** — `valhalla/t5-small-qa-qg-hl` · [HF](https://huggingface.co/valhalla/t5-small-qa-qg-hl) · [patil-suraj/question_generation](https://github.com/patil-suraj/question_generation)
- **Qwen3** — `Qwen/Qwen3-30B-A3B-Instruct-2507` · [HF](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- **MIRAGE** — Vendeville et al. *A Framework for the Quantitative Evaluation of Hallucinations in RAG.* [repo](https://github.com/bVendeville/MIRAGE)
- **FactCC** — Kryściński et al. *Evaluating the Factual Consistency of Abstractive Text Summarization.* EMNLP 2020. [model](https://huggingface.co/manueldeprada/FactCC)
- **SBERT** — Reimers & Gurevych. *Sentence-BERT.* EMNLP 2019. [sentence-transformers](https://www.sbert.net/)
- **Cross-encoder** — `cross-encoder/ms-marco-MiniLM-L-6-v2` (MS MARCO passage reranking)
- **RAG** — Lewis et al. NeurIPS 2020. · **GraphRAG** — Edge et al. 2024 ([arXiv:2404.16130](https://arxiv.org/abs/2404.16130)) · **LLMs + KGs** — Pan et al. 2024 ([arXiv:2306.08302](https://arxiv.org/abs/2306.08302))
- **BM25** — Robertson & Zaragoza, 2009. · **RRF** — Cormack et al. SIGIR 2009.
- **Multi-hop reasoning** — IRCoT (Trivedi et al. ACL 2023) · ReAct (Yao et al. ICLR 2023)

---

## AI usage disclosure

A large language model (GPT-4o) was used as the **generator** for graph-based
question generation in Task 1. Its use is part of the experimental methodology and
is described in the report. The accompanying report (`raport.tex`) is the
authoritative description of the methods and results.
</content>
</invoke>
