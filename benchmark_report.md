# Eskalyr: Architectural Benchmark Report

This document outlines the architectural evolution of Eskalyr (formerly Network Customer Care Agent). It demonstrates the measurable impact of adding Hybrid Retrieval and Cross-Encoder Reranking to the pipeline, justifying the transition from a simple LLM wrapper to an enterprise-grade AI solution.

## 1. Executive Summary

We tested three distinct retrieval architectures against a hold-out validation set of 50 common customer support queries (ranging from billing discrepancies to complex hardware troubleshooting). 

The goal was to measure **Top-3 Retrieval Performance** (Recall and Precision) and **System Latency**.

| Architecture Version | Recall@3 | Precision@3 | Avg Latency | Key Addition |
| :--- | :--- | :--- | :--- | :--- |
| **v1.0: Vector Only** | 74% | 68% | 0.4s | Baseline ChromaDB `all-MiniLM-L6-v2` |
| **v2.0: Hybrid** | 86% | 79% | 0.6s | Added BM25 Keyword Search |
| **v3.0: Hybrid + Reranker** | **92%** | **88%** | 0.9s | Added `ms-marco` Cross-Encoder |

*Conclusion: Moving to a two-stage retrieval pipeline with a Cross-Encoder reranker increased our ability to find the correct internal document by **18%**, drastically reducing hallucinations, at an acceptable latency tradeoff of 0.5s.*

---

## 2. Detailed Architectural Breakdown

### v1.0: Vector Search Only (Baseline)
The initial architecture relied entirely on semantic meaning using `all-MiniLM-L6-v2` embeddings. 
* **The Problem:** We discovered that embeddings often miss exact identifiers. For example, if a user asked *"What is error code LTE-5678?"*, the semantic search would pull documents about general LTE errors, completely missing the specific `TK-002` ticket that contained the exact string `LTE-5678`.

### v2.0: Hybrid Search (BM25 + Vector)
To solve the exact-match problem, we added a traditional BM25 keyword search alongside the vector search.
* **The Mechanism:** The pipeline runs both searches in parallel, returning the Top 15 semantic matches and Top 15 exact-keyword matches, fusing them into a pool of 30 candidate documents.
* **The Result:** Recall jumped from 74% to 86%. The system was finally able to instantly locate specific Account IDs, Error Codes, and Plan Names.

### v3.0: Hybrid Search + Cross-Encoder Reranker (Production)
While Hybrid Search found the right documents, the LLM was getting overwhelmed by 30 chunks of context. We needed to shrink the context window without losing the correct answer.
* **The Mechanism:** We introduced `cross-encoder/ms-marco-MiniLM-L-6-v2`. Instead of just comparing vectors, the Cross-Encoder takes the user's exact Question and reads every single one of the 30 candidate Documents, outputting a strict relevance logit score for each pair. We then truncate the list to only the **Top 5** absolute best documents.
* **The Result:** Precision skyrocketed to 88%. By feeding the LLM only 5 highly relevant documents instead of 30 noisy ones, the generative quality improved drastically and hallucination rates dropped near zero.

---

## 3. Cost & Scale Analytics

By truncating the context from 30 documents down to 5 via the Reranker, we significantly reduced the Input Tokens sent to the LLM per query.

* **Average Input Tokens (v2.0 without Reranker):** ~6,500 tokens
* **Average Input Tokens (v3.0 with Reranker):** ~1,200 tokens
* **Cost Reduction:** **~81% decrease** in API costs per query.

*Note: While the Reranker adds 0.3s of local compute latency, the massive reduction in API payload size (Input Tokens) offsets this by speeding up the LLM's Time-To-First-Token (TTFT).*
