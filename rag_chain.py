import os
import sqlite3
import uuid
import datetime
import math
import time
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 1. Setup the Generator LLM Prompt
SYSTEM_PROMPT = """You are an enterprise network support AI.
Your job is to help customers resolve technical issues.

Use ONLY the context below. Do NOT use outside knowledge.
For every claim you make, explicitly cite the source at the end of the sentence (e.g., "[Ticket ID: TK-002]", "[FAQ ID: 1]", "[Guide Page: 3]").

Context:
{context}
"""

# 2. Setup the Hallucination Verifier Prompt
VERIFIER_PROMPT = """You are a strict compliance verifier.
Read the Context and the AI's Answer.
Can EVERY SINGLE CLAIM in the AI's Answer be explicitly traced back to the Context?

Answer exactly and only "YES" or "NO".

Context:
{context}

AI Answer:
{answer}
"""

def _format_docs(docs: list[Document]) -> str:
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown").lower()
        if source == "faq":
            header = f"[FAQ ID: {doc.metadata.get('faq_id', 'unknown')}]"
        elif source == "ticket":
            header = f"[Ticket ID: {doc.metadata.get('ticket_id', 'unknown')}]"
        elif source == "guide":
            page = doc.metadata.get("page_label") or (doc.metadata.get("page", 0) + 1 if isinstance(doc.metadata.get("page"), int) else doc.metadata.get("page"))
            header = f"[Guide Page: {page}]"
        else:
            header = f"[Source: {source.upper()}]"
        sections.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)

def _get_all_docs_from_chroma(store: Chroma) -> list[Document]:
    data = store.get()
    docs = []
    if "documents" in data and "metadatas" in data:
        for txt, meta in zip(data["documents"], data["metadatas"]):
            docs.append(Document(page_content=txt, metadata=meta))
    return docs

def create_support_ticket(question: str) -> str:
    ticket_id = f"TK-{uuid.uuid4().hex[:4].upper()}"
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect('data/escalations.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS escalations (id TEXT, question TEXT, date TEXT, status TEXT)''')
    c.execute('INSERT INTO escalations VALUES (?, ?, ?, ?)', (ticket_id, question, str(datetime.datetime.now()), "Open"))
    conn.commit()
    conn.close()
    return ticket_id

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def build_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

    faq_store = Chroma(collection_name="faq", embedding_function=embeddings, persist_directory=CHROMA_DIR)
    tickets_store = Chroma(collection_name="tickets", embedding_function=embeddings, persist_directory=CHROMA_DIR)
    guides_store = Chroma(collection_name="guides", embedding_function=embeddings, persist_directory=CHROMA_DIR)

    all_docs = []
    for store in [faq_store, tickets_store, guides_store]:
        all_docs.extend(_get_all_docs_from_chroma(store))
        
    bm25_retriever = BM25Retriever.from_documents(all_docs) if all_docs else None
    if bm25_retriever:
        bm25_retriever.k = 15

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    
    verifier_prompt = ChatPromptTemplate.from_template(VERIFIER_PROMPT)
    
    model_name = "qwen/qwen3-32b"
    llm = ChatGroq(model=model_name, temperature=0, max_tokens=None)

    def rag_pipeline(inputs: dict):
        start_time = time.time()
        question = inputs.get("question", "")
        chat_history = inputs.get("chat_history", [])
        
        # Base cost tracking variables
        input_tokens = 0
        output_tokens = 0
        cost = 0.0

        v_docs = []
        for store in [faq_store, tickets_store, guides_store]:
            results = store.similarity_search(question, k=5)
            v_docs.extend(results)
            
        k_docs = bm25_retriever.invoke(question) if bm25_retriever else []
        
        unique_docs = {}
        for d in v_docs + k_docs:
            unique_docs[d.page_content] = d
        candidate_docs = list(unique_docs.values())

        if not candidate_docs:
            return {
                "answer": "I could not find any information to help with that.",
                "confidence": 0.0,
                "escalated": False,
                "docs": [],
                "latency": time.time() - start_time,
                "input_tokens": 0, "output_tokens": 0, "cost": 0.0, "model": model_name
            }

        pairs = [[question, doc.page_content] for doc in candidate_docs]
        scores = reranker.predict(pairs)
        
        scored_docs = list(zip(candidate_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_5 = scored_docs[:5]
        
        raw_scores = [score for _, score in top_5]
        # Use the highest score to determine confidence, not the average! 
        # (Because you only need 1 good document to answer a question)
        best_logit = max(raw_scores) if raw_scores else -10.0
        confidence_pct = sigmoid(best_logit) * 100
        
        final_docs = [doc for doc, _ in top_5]
        doc_sources = [doc.metadata.get("source", "unknown") for doc in final_docs]
        formatted_context = _format_docs(final_docs)

        if confidence_pct < 30.0:
            ticket_id = create_support_ticket(question)
            ans = f"**Confidence: Very Low ({confidence_pct:.1f}%)**\n\nI could not confidently answer this question from the knowledge base.\n\n**Action Taken:** A support ticket has been automatically created for a human agent.\n**Ticket ID:** `{ticket_id}`\n**Priority:** High"
            return {
                "answer": ans, "confidence": confidence_pct, "escalated": True, "docs": doc_sources,
                "latency": time.time() - start_time, "input_tokens": 0, "output_tokens": 0, "cost": 0.0, "model": model_name
            }
            
        warning = f"**Confidence: Low ({confidence_pct:.1f}%)** - *Found some related information, but none of the retrieved documents may directly answer the question.*\n\n---\n\n" if confidence_pct < 65.0 else f"**Confidence: High ({confidence_pct:.1f}%)**\n\n---\n\n"

        # Track prompt length for cost approximation
        prompt_val = prompt.invoke({"context": formatted_context, "chat_history": chat_history, "question": question})
        prompt_text = prompt_val.to_string()
        input_tokens += len(prompt_text.split()) * 1.3 
        
        answer = llm.invoke(prompt_val).content
        output_tokens += len(answer.split()) * 1.3

        # Clean up <think> tags from reasoning models
        import re
        clean_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        
        # Verifier step
        ver_val = verifier_prompt.invoke({"context": formatted_context, "answer": clean_answer})
        input_tokens += len(ver_val.to_string().split()) * 1.3
        
        verification_result = llm.invoke(ver_val).content.strip().upper()
        output_tokens += 5 # Verifier typically responds with YES/NO
        
        # Calculate mock API cost ($0.50 per 1M input, $0.80 per 1M output)
        cost = (input_tokens / 1_000_000 * 0.50) + (output_tokens / 1_000_000 * 0.80)
        
        # Check if the Verifier explicitly said NO at the very beginning of its response
        if verification_result.startswith("NO"):
            ticket_id = create_support_ticket(question)
            ans = warning + "**Safety Guardrail Triggered:**\nAn answer was generated, but the internal verifier detected that not every claim could be explicitly traced back to the official documentation. To prevent hallucinations, the answer was suppressed.\n\n" + f"A support ticket has been created: `{ticket_id}`"
            return {
                "answer": ans, "confidence": confidence_pct, "escalated": True, "docs": doc_sources,
                "latency": time.time() - start_time, "input_tokens": int(input_tokens), "output_tokens": int(output_tokens), "cost": cost, "model": model_name
            }

        return {
            "answer": warning + clean_answer, "confidence": confidence_pct, "escalated": False, "docs": doc_sources,
            "latency": time.time() - start_time, "input_tokens": int(input_tokens), "output_tokens": int(output_tokens), "cost": cost, "model": model_name
        }

    return RunnableLambda(rag_pipeline)
