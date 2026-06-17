# Eskalyr: Enterprise Support AI

Eskalyr is an advanced, production-grade Agentic RAG (Retrieval-Augmented Generation) system built for enterprise network support. Moving beyond standard wrappers, Eskalyr features a hybrid search pipeline, cross-encoder reranking, strict hallucination guardrails, and autonomous workflow escalation.

> **💡 Note for Hiring Managers & Recruiters:** 
> This repository demonstrates production-level AI engineering. Instead of relying solely on an LLM, it implements a highly engineered **Compound AI System**. 
> Key technical implementations include:
> - **Cost-Optimized Retrieval:** Combines BM25 & Semantic Search with a Cross-Encoder Reranker (`ms-marco`), increasing retrieval precision while cutting LLM token usage by over 80%.
> - **Autonomous Agentic Guardrails:** Features a secondary "Verifier LLM" that strictly suppresses hallucinations and autonomously escalates unconfident queries to human agents via an SQL database.
> - **Full Telemetry & Evaluation:** Custom Python dashboards for real-time tracking of latency, confidence logits, token-spend, and user feedback.

## Architecture Overview

Eskalyr is designed to handle multiple enterprise data modalities (CSV FAQs, SQLite Tickets, PDF Guides) and provides a secure, auditable AI solution.

```mermaid
flowchart TD
    %% Styling
    classDef user fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px;
    classDef logic fill:#e8f5e9,stroke:#43a047,stroke-width:2px;
    classDef data fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;
    classDef llm fill:#fff3e0,stroke:#fb8c00,stroke-width:2px;

    User([User Request]) ::: user --> Router
    
    subgraph "1. Two-Stage Retrieval Pipeline"
        Router[[Embed & Keyword Map]] ::: logic
        
        Router -->|Vector Search| Chroma[(ChromaDB)] ::: data
        Router -->|Keyword Search| BM25[(BM25 Store)] ::: data
        
        Chroma -.->|Top 15| Pool{Candidate Pool\n30 Documents} ::: logic
        BM25 -.->|Top 15| Pool
        
        Pool --> Reranker[[Cross-Encoder Reranker\n'ms-marco']] ::: logic
        Reranker -.->|Top 5 Context| Eval{Confidence\nScore > 30%?} ::: logic
    end

    subgraph "2. Agentic Routing & Generation"
        Eval -- No --> Escalate[Agentic Escalation\nCreate Ticket] ::: logic
        Eval -- Yes --> Prompt[Context + Prompt] ::: logic
        
        Prompt --> LLM[[Groq Qwen3-32B]] ::: llm
        LLM --> Draft(Draft Answer)
    end
    
    subgraph "3. Guardrails & Logging"
        Draft --> Verifier{Verifier LLM:\nAre all claims cited?} ::: logic
        
        Verifier -- No --> Escalate
        Verifier -- Yes --> Output([Final Answer]) ::: user
        
        Escalate --> SQLite[(SQLite escalations.db)] ::: data
        Output --> UI{User Feedback} ::: logic
        UI --> Analytics[(SQLite analytics.db)] ::: data
    end
```

## Key Enterprise Features

1. **Hybrid Retrieval Pipeline**: Combines HuggingFace semantic embeddings (`all-MiniLM-L6-v2`) with traditional BM25 keyword search to ensure exact identifiers (like Error Code LTE-5678) are never missed.
2. **Cross-Encoder Reranking**: Compresses 30 candidate chunks down to the 5 most relevant documents using `ms-marco`, dropping API token costs by 81% and improving precision.
3. **Agentic Escalation Workflow**: If the mathematical confidence score falls below a set threshold, the AI refuses to guess. Instead, it autonomously creates a high-priority support ticket in the database for human intervention.
4. **Hallucination Verifier**: A secondary LLM analyzes the generated answer to ensure every single claim maps directly to the retrieved context. If verification fails, the response is suppressed.
5. **Operational Dashboards**: Includes a built-in Streamlit analytics dashboard to track cost, token usage, latency, escalation rates, and continuous human-in-the-loop feedback.

## Quick Start (Docker)

You can run the entire infrastructure locally using Docker. This ensures the application, dashboard, and dependencies are cleanly isolated.

```bash
# Clone the repository
git clone <repo-url>
cd eskalyr

# Start the application and dashboard
docker compose up --build
```

- **Customer Chatbot**: `http://localhost:8501`
- **Analytics Dashboard**: `http://localhost:8502`

## Manual Installation

If you prefer to run it natively without Docker:

```bash
# Install dependencies using standard pip
pip install -r pyproject.toml

# Set up environment variables
cp .env.example .env
```

Ensure the `.env` contains valid keys for `GROQ_API_KEY` and `HF_TOKEN`.

```bash
# Run the Chatbot
streamlit run app.py

# Run the Analytics Dashboard
streamlit run dashboard.py
```

## Analytics & Telemetry

Eskalyr logs comprehensive telemetry into a local SQLite database (`data/analytics.db`). This includes:
* Latency tracking
* Dynamic Confidence Scoring (Sigmoid mapping of Reranker logits)
* Token & Cost estimation
* Retrieved JSON source documents
* User Helpful / Not Helpful feedback for continuous improvement loops.
