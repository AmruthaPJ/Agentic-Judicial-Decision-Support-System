# ⚖️ Judicial Decision Support System (JDSS)
### *Agentic AI-Driven Architecture for Docket Triage, IPC $\leftrightarrow$ BNS Statutory Translation, and Corrective Precedent Retrieval*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6F00.svg?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Legal--BERT-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1.svg?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![XGBoost & SHAP](https://img.shields.io/badge/XAI-XGBoost%20%2B%20TreeSHAP-red.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Academic%20Research-green.svg)]()

---

## 📌 Executive Overview

The Indian judicial system faces a systemic pendency crisis, with over **40 million cases pending** across district courts, High Courts, and the Supreme Court. Traditional case listing largely operates on unranked chronological order (**First-In, First-Out / FIFO**) or ad-hoc listing practices. Under this regime, routine procedural applications and boundary disputes can unintentionally crowd out time-sensitive life-and-liberty matters such as bail petitions, habeas corpus pleas, and urgent injunctions. Furthermore, the Indian legal system is navigating a historic statutory transition from the colonial **Indian Penal Code (IPC 1860)** and **Code of Criminal Procedure (CrPC 1973)** to the newly enacted **Bharatiya Nyaya Sanhita (BNS 2023)** and **Bharatiya Nagarik Suraksha Sanhita (BNSS 2023)**.

The **Judicial Decision Support System (JDSS)** is an explainable, human-in-the-loop multi-agent AI architecture designed to tackle both challenges. JDSS integrates:
1. **Statutory Bridge:** Real-time bidirectional translation between IPC, CrPC, and BNS provisions.
2. **Corrective Precedent Retrieval (CRAG):** Hybrid semantic search combining lexical BM25, Legal-BERT transformer embeddings, Reciprocal Rank Fusion (RRF), and a self-correcting LangGraph retrieval agent.
3. **Statutory Docket Prioritizer:** Multi-factor urgency scoring balancing offense severity, fundamental liberty threats, societal impact, and progressive aging escalation.
4. **Explainable AI (XAI):** TreeSHAP feature attributions providing interpretable decision justifications for presiding judges.
5. **Algorithmic Fairness & Demographic Language Screening:** Dual LLM-as-a-judge and rule-based screening to detect demographic bias and profiling.
6. **LangGraph Judicial Reconciliation Agent:** An evaluator-optimizer workflow that intercepts conflicting dossiers, enforces deterministic liberty score floors, and synthesizes advisory triage memos.

> **Disclaimer:** JDSS is strictly an advisory, human-in-the-loop **Decision Support System**. It is not an autonomous judge. All outputs, rankings, and memos serve to assist registry masters and presiding judicial officers in docket administration.

---

## 🏛️ System Architecture

The end-to-end multi-agent pipeline is orchestrated via **LangGraph**, backed by **PostgreSQL (`pgvector`)**, **Legal-BERT**, **Gemini 2.5 Flash**, and served through a modern **FastAPI** backend and **React** UI:

```mermaid
flowchart TD
    subgraph Ingestion["1. Ingestion & Preprocessing"]
        PDF["Case PDF / Judgment"] --> TextExtract["PyMuPDF Text & Metadata Extraction"]
        TextExtract --> Chunker["Legal Boundary Chunking"]
        TextExtract --> SectionParser["Regex Statutory Parser (IPC / BNS / CPC)"]
    end

    subgraph StatuteBridge["2. Statutory Cross-Regime Bridge"]
        SectionParser --> MappingTable["IPC ↔ BNS Bidirectional Translation Table"]
        MappingTable --> SeverityWeight["Statutory Offence Severity Assignment (1-10)"]
    end

    subgraph RetrievalEngine["3. Corrective Precedent Retrieval (CRAG)"]
        Chunker --> Embeddings["Legal-BERT Embeddings (768-dim)"]
        Embeddings --> PGVector[("PostgreSQL + pgvector")]
        Lexical["BM25 Lexical Index"] --> RRF["Reciprocal Rank Fusion (RRF)"]
        PGVector --> RRF
        RRF --> LG_Retrieve{"LangGraph CRAG Agent"}
        LG_Retrieve -->|Score < 0.65| Rewrite["Query Rewriting & Refinement"]
        Rewrite --> LG_Retrieve
        LG_Retrieve -->|Verified Precedents| Dossier["Case Dossier Aggregator"]
    end

    subgraph Prioritization["4. Priority & Explainability Engine"]
        SeverityWeight --> ScoreCalc["Composite Statutory Formulation"]
        TextExtract --> ThreatCheck["Liberty / Immediate Threat Detector"]
        TextExtract --> ImpactCheck["Societal Impact Detector (PIL / Constitution)"]
        TextExtract --> AgeEscalation["Progressive Aging Escalation Factor"]
        ThreatCheck --> ScoreCalc
        ImpactCheck --> ScoreCalc
        AgeEscalation --> ScoreCalc
        ScoreCalc --> XGBoost["XGBoost Surrogate Regressor"]
        XGBoost --> TreeSHAP["TreeSHAP Explainability Attributions"]
        TreeSHAP --> Dossier
    end

    subgraph Governance["5. Fairness & Reconciliation Guardrails"]
        TextExtract --> BiasAudit["Demographic Language Screen (LLM / Lexical)"]
        BiasAudit --> Dossier
        Dossier --> Reconciler{"LangGraph Reconciliation Agent"}
        Reconciler -->|Liberty Exception or Low Score| Deliberation["Gemini Deliberation & Floor Guardrail (≥ 6.0)"]
        Reconciler -->|Standard Approval| Approved["Approved Dossier"]
        Deliberation --> TriageMemo["Advisory Triage Memo (FLAGGED_FOR_BENCH)"]
    end

    subgraph Presentation["6. Judicial Bench Interface"]
        Approved --> BenchUI["React Judicial Workbench & Streamlit Dashboard"]
        TriageMemo --> BenchUI
    end
```

---

## 🧩 Core Subsystems & Technical Implementation

### 1. Statutory Bridge (IPC $\leftrightarrow$ BNS Translation)
* **Objective:** Enable courts to search 75+ years of historical jurisprudence decided under the Indian Penal Code (1860) using modern Bharatiya Nyaya Sanhita (2023) terminology, and vice-versa.
* **Mechanism:** Normalized dictionary mapping cross-referencing chapters, sections, legal ingredients, and baseline penal severity weights ($1$ to $10$).
* **Fuzzy Parsing:** Normalizes ambiguous legal notations (e.g., `S. 302`, `Section 302 IPC`, `304-B`, `Section 103 BNS`).

### 2. Corrective Precedent Retrieval (LangGraph CRAG)
* **Dense Semantic Search:** Generates 768-dimensional sentence-level representations using [`nlpaueb/legal-bert-base-uncased`](https://huggingface.co/nlpaueb/legal-bert-base-uncased), indexed using HNSW/IVFFlat cosine distance in `pgvector`.
* **Sparse Lexical Search:** BM25Okapi scoring across normalized legal corpora for exact statutory citations.
* **Reciprocal Rank Fusion (RRF):** Fuses ranking lists to eliminate retriever bias:
  $$\text{RRF Score}(d) = \sum_{m \in \{\text{BM25}, \text{BERT}\}} \frac{1}{k + \text{rank}_m(d)}, \quad k = 60$$
* **LangGraph Agent Workflow:** Employs an iterative StateGraph (`grade_first_pass` $\rightarrow$ `rewrite_query` $\rightarrow$ `retrieve_rewritten` $\rightarrow$ `grade_retry`) ensuring retrieved citations exceed strict relevance thresholds before reaching the judge.

### 3. Statutory Docket Prioritization Formulation
The composite priority score $P \in [1.0, 10.0]$ is grounded in High Court and Supreme Court Practice Rules:
$$P = (S_{\text{norm}} \times 3.5) + (T_{\text{norm}} \times 3.0) + (I_{\text{norm}} \times 2.5) + E_{\text{age}} + \Delta_{\text{listing}}$$

Where:
* **$S_{\text{norm}} = S / 10$:** Offense severity rating ($S \in [1, 10]$). Civil matters are capped at $0.45$ to prevent commercial disputes from crowding out criminal justice.
* **$T_{\text{norm}} \in \{0, 1\}$:** Immediate Threat Flag triggered by urgent liberty restraints (regular/anticipatory bail, habeas corpus, custodial torture, demolition stays, death sentence reviews).
* **$I_{\text{norm}} = I / 5$:** Societal Impact Score ($I \in [1, 5]$) reflecting Public Interest Litigations (PILs), fundamental rights under Art. 32/226, environmental hazards, or widespread financial scams.
* **$E_{\text{age}} = \min(1.30, \max(0.15, 0.15 + (\text{Years} \times 0.065)))$:** Continuous, progressive aging escalation preventing chronic pendency and case starvation.
* **$\Delta_{\text{listing}}$:** Fast-track adjustments recognized by registry listing guidelines:
  * Senior Citizen litigant ($\ge 65$ years): $+0.85$
  * Contempt of Court (Art. 129 / 215): $+1.40$
  * Interlocutory stay/injunction petition: $+0.50$
  * Procedural miscellaneous applications: $-0.45$

### 4. Explainable AI (XAI) with TreeSHAP
* Presiding judges cannot rely on unexplainable "black-box" predictions.
* An **XGBoost regressor** acts as a sub-millisecond surrogate model approximating the multi-factor heuristic for high-throughput docket ingestion.
* **TreeSHAP** computes exact local Shapley attributions for every case, explaining the marginal contribution of each fact (e.g. *"+3.1 due to bail petition, +1.2 due to murder charge under IPC 302, -0.4 due to short pendency"*).

### 5. Fairness Auditing & Demographic Language Screen
* **LLM-as-a-Judge Audit:** Evaluates case judgments and FIR charge-sheets via Gemini 2.5 Flash for prejudicial profiling, caste/religious profiling, or discriminatory sentencing remarks.
* **Lexical Rule-Based Fallback:** Detects sensitive profiling markers if offline or without API connectivity.
* **Counterfactual Invariance:** Ensures perturbation of demographic variables (e.g. name, religion, caste) while holding legal facts constant produces zero delta in priority score ($\Delta P = 0.0$).

### 6. LangGraph Judicial Reconciliation Agent
An Evaluator-Optimizer StateGraph acting as a final safeguard:
1. **Audit Dossier Node:** Compares the computed priority against fundamental liberty signals. If an urgent liberty signal exists (e.g. habeas corpus) but the priority score is $< 6.0$, a conflict is flagged.
2. **Conditional Router:** Bypasses normal approval and diverts conflicting cases to deliberation.
3. **Deliberation Node:** Enforces a deterministic safety floor ($P \ge 6.0$) and synthesizes an advisory **Judicial Triage Memo** marked `FLAGGED_FOR_BENCH` for human review.

---

## 📊 Empirical Evaluation & Benchmark Alignment

The system was evaluated across six experimental dimensions:

### 1. Courtroom Discrete-Event Simulation
* **Setup:** Modeled after an Indian High Court with 3 virtual benches, a throughput of **6 cases/day**, **500 initial backlog cases**, and **180 simulated days** evaluated across **30 independent Poisson seeds** under Below (2/day), Near (6/day), and Above (8/day) capacity workloads.
* **Results:**
  * **Critical / Urgent Cases Turnaround:** Reduced from **83.3 days** under FIFO to **19.4 days** under AI Prioritization (a **~74.5% acceleration**).
  * **Systemic Trade-Off Analysis:** Demonstrates that prioritizing urgent life/liberty cases appropriately stretches the queue for low-severity civil property disputes during heavy docket surges—an intentional, human-centered policy trade-off.

### 2. Precedent Retrieval Benchmark
* **Setup:** Evaluated 13 diverse legal domains (IPC homicide, rape, habeas corpus, CPC injunctions, environmental PILs, and cross-regime BNS queries).
* **Finding:** Lexical search (BM25) and dense semantic search (Legal-BERT) exhibit only a **20% to 40% Overlap@10**, confirming that neither method alone suffices. The **RRF Hybrid** retriever effectively captures both exact statutory references and semantic legal synonyms.

### 3. Summarization Fidelity
* Evaluated against case corpora using `rouge-score`:
  * **ROUGE-1 F1:** `0.3883` | **ROUGE-2 F1:** `0.2101` | **ROUGE-L F1:** `0.1872`
  * **Mean Compression Ratio:** `17.4%` (compresses ~5,000 words down to ~870 words).
  * **Statutory Section Preservation:** `100.0%` (verified through regex-bounded statutory citation matching).

### 4. Global SHAP Feature Importance
TreeSHAP attributions across all 863 cases confirm that the model bases decisions strictly on legally material factors:
1. **Immediate Threat Flag:** $+1.1876$ (Dominant driver)
2. **Offence Severity Rating:** $+0.3949$
3. **Case Pending Days:** $+0.2964$
4. **Societal Impact Score:** $+0.2832$
5. *Incidental features (precedent count, word count, docket format):* $< 0.003$

### 5. Inter-Rater Reliability & Ranking Consistency
* **Spearman Rank Correlation ($\rho$):** `0.908` ($p < 0.001$, extremely strong monotonic ranking agreement).
* **Kendall’s Tau ($\tau$):** `0.811` (over 90% pairwise relative ordering concordance).
* **Mean Absolute Score Error (MAE):** `0.202 / 10` ($\approx 2\%$ deviation).
* **Priority-Band Triage Agreement:** `96.4%` (Critical, High, Medium, Low category match).

### 6. Indian Legal NLP Benchmark Alignment
JDSS directly aligns with recognized academic benchmarks for Indian law:
* **[IL-TUR (ACL 2024)](https://github.com/Exploration-Lab/IL-TUR):** Benchmark for Indian Legal Text Understanding and Reasoning (aligns with Prior Case Retrieval [PCR], Legal Statute Identification [LSI], and Summarization [SUMM]).
* **[IL-PCSR (EMNLP 2025)](https://github.com/Exploration-Lab/IL-PCSR):** Joint benchmark for Indian Legal Prior Case and Statute Retrieval.
* **[IN-Abs & IN-Ext](https://github.com/Law-AI/summarization):** Standard Indian Supreme Court abstractive and extractive summarization datasets.

---

## 📂 Project Directory Structure

```text
Agentic-Judicial-Decision-Support-System/
├── backend/
│   └── api.py                     # FastAPI REST API server
├── frontend/
│   ├── src/                       # React 19 + TypeScript components
│   ├── package.json               # Frontend dependencies & scripts
│   └── vite.config.ts             # Vite bundler configuration
├── agents/
│   ├── corrective_precedent_search.py  # LangGraph CRAG agent
│   ├── gemini_client.py           # Google Gemini API connector
│   ├── fairness/
│   │   ├── bias_agent.py          # LLM & lexical fairness screen
│   │   └── bias_detector.py       # Demographic profiling rules
│   ├── prioritization/
│   │   ├── prioritizer.py         # Priority engine module
│   │   └── scoring.py             # Statutory formulation & listing rules
│   ├── explainability/
│   │   └── explainability_agent.py# TreeSHAP attribution generator
│   ├── reconciliation/
│   │   └── reconciliation_agent.py# LangGraph Evaluator-Optimizer
│   └── summarization/
│       └── summarizer.py          # Legal case document summarizer
├── data/
│   ├── chunks/                    # Preprocessed JSON chunks per case
│   ├── embeddings/                # Vector representations (Legal-BERT)
│   ├── mappings/                  # IPC ↔ BNS statutory translation maps
│   └── prioritization/            # Feature CSVs, models, and encoders
├── evaluation/
│   ├── simulate_courtroom.py      # Discrete-event 30-seed simulation
│   ├── benchmark_retrieval.py     # BM25 vs Legal-BERT vs RRF benchmark
│   ├── evaluate_rouge.py          # ROUGE & statutory section fidelity
│   ├── generate_shap_plots.py     # TreeSHAP beeswarm & waterfall plots
│   └── analyze_expert_agreement.py# Spearman, Kendall, & Kappa evaluation
├── scripts/
│   ├── pdf_to_text.py             # PyMuPDF document ingestion
│   ├── chunk_text.py              # Semantic text boundary chunking
│   ├── generate_embeddings.py     # Batch Legal-BERT embedding generation
│   ├── migrate_to_pg.py           # PostgreSQL + pgvector schema setup
│   └── train_prioritizer.py       # XGBoost surrogate training
├── paper/
│   └── Judicial_AI_System_Paper.tex # Formal research publication draft
├── config.py                      # Centralized configuration & DB URLs
├── docker-compose.yml             # PostgreSQL 16 + pgvector container
└── requirements.txt               # Python package dependencies
```

---

## 🚀 Quickstart & Installation Guide

### Prerequisites
* **Python:** 3.10 or 3.11
* **Node.js:** 18+ and `npm`
* **Docker & Docker Compose:** For running the `pgvector` database
* **Google Gemini API Key:** For LLM fairness screening and reconciliation

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/AmruthaPJ/Agentic-Judicial-Decision-Support-System.git
cd Agentic-Judicial-Decision-Support-System
```

### Step 2: Configure Environment Variables
Create a `.env` file in the root directory:
```env
# Database Connection
DATABASE_URL=postgresql://admin:password@localhost:5432/judicial

# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 3: Start the Vector Database
Launch the PostgreSQL instance pre-configured with `pgvector`:
```bash
docker compose up -d
```
*Verify it is running on port `5432`:*
```bash
docker ps
```

### Step 4: Set Up Python Backend Environment
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 5: Initialize Database Schema & Ingestion Pipeline
```bash
# 1. Initialize PostgreSQL tables and pgvector extension
python scripts/migrate_to_pg.py

# 2. Extract statutory features and calculate priority scores
python scripts/extract_features.py
python scripts/create_labels.py

# 3. Generate Legal-BERT vector embeddings for all cases
python scripts/generate_embeddings.py

# 4. Train the explainable XGBoost surrogate prioritizer
python scripts/train_prioritizer.py
```

### Step 6: Launch the Backend API
Start the FastAPI server:
```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```
*Access interactive Swagger API docs at:* `http://localhost:8000/docs`

### Step 7: Launch the Frontend Interface
In a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
*Open your browser at:* `http://localhost:5173`

---

## 🧪 Running the Evaluation Suite

To independently run and verify all empirical evaluations:

```bash
# 1. Run the 30-seed discrete-event courtroom simulation
python evaluation/simulate_courtroom.py

# 2. Run the retrieval benchmark (BM25 vs Legal-BERT vs RRF)
python evaluation/benchmark_retrieval.py

# 3. Evaluate summarization ROUGE scores & statutory preservation
python evaluation/evaluate_rouge.py

# 4. Generate TreeSHAP beeswarm and feature importance plots
python evaluation/generate_shap_plots.py

# 5. Run expert inter-rater agreement analysis
python evaluation/analyze_expert_agreement.py
```
All evaluation tables (`.tex` and `.csv`) and visual plots will be generated in the `evaluation/` directory.

---

## 🛡️ Ethical Governance & Safety Guardrails

1. **Human-in-the-Loop Principle:** The system generates advisory triage recommendations; it does not replace judicial discretion. Final docket scheduling remains under the authority of the presiding registry.
2. **Deterministic Liberty Floor:** The LangGraph Reconciliation Agent enforces a non-negotiable minimum score of $6.0$ for any petition involving fundamental liberty (habeas corpus, custody, unlawful detention), overriding any model under-prediction.
3. **Audit Trails & Explainability:** Every priority rating includes a human-readable TreeSHAP breakdown of contributing factors to maintain complete transparency and accountability.

---

## 👥 Authors & Project Information

* **Institution:** Department of Computer Science and Engineering
* **Project Title:** Agentic AI-Driven Judicial Decision Support System
* **Primary Repository:** [AmruthaPJ/Agentic-Judicial-Decision-Support-System](https://github.com/AmruthaPJ/Agentic-Judicial-Decision-Support-System)
* **Branch:** `main`

---
*Developed as a Major Final Year Engineering Project (Project 97).*
