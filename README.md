# ⚖️ Agentic Judicial Decision Support System (JDSS)
### *Multi-Agent AI for Docket Triage, IPC $\leftrightarrow$ BNS Statutory Translation, and Precedent Retrieval*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6F00.svg?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Legal-BERT](https://img.shields.io/badge/PyTorch-Legal--BERT-EE4C2C.svg?logo=pytorch&logoColor=white)](https://huggingface.co/nlpaueb/legal-bert-base-uncased)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1.svg?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![XAI](https://img.shields.io/badge/XAI-XGBoost%20%2B%20TreeSHAP-red.svg)](https://shap.readthedocs.io/)

---

## 📌 Overview

The Indian judicial system faces a massive backlog of over **40 million pending cases**. Unranked chronological listing (**FIFO**) often causes urgent life-and-liberty matters (such as bail and habeas corpus) to wait behind routine procedural disputes. At the same time, Indian courts are navigating the statutory transition from the **Indian Penal Code (IPC 1860)** to the **Bharatiya Nyaya Sanhita (BNS 2023)**.

**JDSS** is an explainable, human-in-the-loop multi-agent AI system designed to assist registry masters and judges:
* **Statutory Bridge:** Real-time bidirectional translation between IPC, CrPC, and modern BNS provisions.
* **Corrective Precedent Retrieval (CRAG):** Hybrid semantic search fusing BM25, Legal-BERT embeddings, Reciprocal Rank Fusion (RRF), and LangGraph self-evaluation.
* **Explainable Docket Triage:** Urgency scoring based on offense gravity, liberty threats, and aging escalation, explained via **TreeSHAP**.
* **Safety & Reconciliation Guardrails:** A LangGraph evaluator-optimizer agent enforcing deterministic score floors ($\ge 6.0$) on liberty cases with demographic bias screening.

> **Note:** JDSS is an advisory decision support tool for court registries and judges, not an autonomous adjudicator.

---

## 🔄 End-to-End System Workflow

```mermaid
flowchart LR
    In["📄 1. Ingestion\nCase Judgment / Petition PDF"] --> Bridge["⚖️ 2. Statutory Bridge\nIPC ↔ BNS & CrPC ↔ BNSS"]
    
    subgraph Agents["3. Multi-Agent Processing"]
        direction TB
        CRAG["Precedent Retrieval\nLegal-BERT + BM25 + RRF"]
        Triage["Docket Prioritizer\nXGBoost + TreeSHAP (0-10)"]
        Screen["Fairness Screening\nGemini 2.5 Flash Bias Audit"]
    end
    
    Bridge --> Agents
    
    Agents --> Guard["🛡️ 4. Reconciliation Guardrail\nLangGraph StateGraph\n(Enforces Liberty Floor ≥ 6.0)"]
    
    Guard --> UI["🖥️ 5. Judicial Bench\nAdvisory Triage Memo\n& Interactive React UI"]
```

---

## ⚡ Key Capabilities

| Capability | Technology | Description |
| :--- | :--- | :--- |
| **IPC $\leftrightarrow$ BNS Translation** | Rule-Based Parser + Mapping Matrix | Cross-references colonial sections with modern penal provisions and baseline severities. |
| **Precedent Search (CRAG)** | Legal-BERT (768-d), `pgvector`, BM25, RRF | Hybrid dense-sparse retrieval with iterative LangGraph query reformulation. |
| **Case Prioritization** | Multi-Factor Heuristic + XGBoost | Balances statutory severity ($35\%$), liberty threat ($30\%$), public impact ($25\%$), and pendency age. |
| **Explainable AI (XAI)** | TreeSHAP | Provides local waterfall feature attributions so judges see *why* a case received its priority. |
| **Safety Guardrail** | LangGraph StateGraph | Intercepts cases where urgent liberty signals clash with lower scores, raising them to $\ge 6.0$. |

---

## 📊 Evaluation Summary

Evaluated on Supreme Court of India case records across empirical and simulation benchmarks:

* **Courtroom Simulation (30 Seeds, 180 Days):**  
  Reduces turnaround time for critical life/liberty cases by **$74.5\%$** (from $83.3$ days under FIFO to $19.4$ days under AI prioritization).
* **Inter-Rater & Ranking Consistency:**  
  * **Spearman Rank Correlation ($\rho$):** `0.908` ($p < 0.001$, strong monotonic agreement).  
  * **Kendall’s Tau ($\tau$):** `0.811` (over $90\%$ pairwise relative ordering consistency).  
  * **Priority-Band Triage Match:** `96.4%` agreement across Critical, High, Medium, and Low tiers.
* **Summarization & Citation Fidelity:**  
  * **ROUGE-1 F1:** `0.388` | **ROUGE-2 F1:** `0.210` | **ROUGE-L F1:** `0.187` (at $17.4\%$ compression).  
  * **Statutory Citation Retention:** `100.0%` precision on section numbers.
* **Benchmark Alignment:** Framed against Indian Legal NLP benchmarks **[IL-TUR (ACL 2024)](https://github.com/Exploration-Lab/IL-TUR)** and **[IL-PCSR (EMNLP 2025)](https://github.com/Exploration-Lab/IL-PCSR)**.

---

## 📂 Project Structure

```text
Agentic-Judicial-Decision-Support-System/
├── backend/                  # FastAPI REST API (search, triage, dossier)
├── frontend/                 # React 19 + TypeScript judicial workbench
├── agents/                   # LangGraph agents (CRAG, Prioritizer, XAI, Reconciler)
├── evaluation/               # Simulation, ROUGE, SHAP, and inter-rater scripts
├── scripts/                  # Ingestion, feature extraction, and ML training pipelines
├── data/                     # Case chunks, embeddings, and IPC-BNS mappings
├── docker-compose.yml        # PostgreSQL + pgvector container
└── requirements.txt          # Python dependencies
```

---

## 🚀 Quickstart

### 1. Prerequisites
* Python 3.10+, Node.js 18+, Docker

### 2. Setup Database & Environment
```bash
# Clone the repository
git clone https://github.com/AmruthaPJ/Agentic-Judicial-Decision-Support-System.git
cd Agentic-Judicial-Decision-Support-System

# Start PostgreSQL with pgvector
docker compose up -d

# Configure .env
echo "DATABASE_URL=postgresql://admin:password@localhost:5432/judicial" > .env
echo "GEMINI_API_KEY=your_gemini_api_key_here" >> .env
```

### 3. Launch Backend
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run FastAPI server
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```
*API Documentation available at:* `http://localhost:8000/docs`

### 4. Launch Frontend
```bash
cd frontend
npm install
npm run dev
```
*Open your browser at:* `http://localhost:5173`

---

## 👥 Authors & Project Information

* **Institution:** Department of Computer Science and Engineering
* **Project:** Agentic AI-Driven Judicial Decision Support System (Major Project 97)
* **Primary Repository:** [AmruthaPJ/Agentic-Judicial-Decision-Support-System](https://github.com/AmruthaPJ/Agentic-Judicial-Decision-Support-System)
