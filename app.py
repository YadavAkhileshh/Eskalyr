import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sqlite3
import datetime
import json
import time
import streamlit as st
from dotenv import load_dotenv
from rag_chain import build_chain

# Load environment variables like API keys
load_dotenv()

# Quick-start questions available for instant click
SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "My calls keep dropping - what should I do?",
    "How do I activate international roaming?",
    "Why is my bill higher than usual this month?",
    "My phone shows SIM not detected after a restart",
]

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect('data/analytics.db')
    c = conn.cursor()
    # Analytics Table logs everything about the request
    c.execute('''CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    question TEXT, 
                    answer TEXT, 
                    confidence REAL,
                    latency REAL,
                    escalated BOOLEAN,
                    docs TEXT,
                    helpful BOOLEAN,
                    date TEXT)''')
    
    # Add new cost tracking columns if they don't exist
    try:
        c.execute("ALTER TABLE analytics ADD COLUMN input_tokens INTEGER")
        c.execute("ALTER TABLE analytics ADD COLUMN output_tokens INTEGER")
        c.execute("ALTER TABLE analytics ADD COLUMN total_cost REAL")
        c.execute("ALTER TABLE analytics ADD COLUMN model_used TEXT")
    except sqlite3.OperationalError:
        pass # Columns already exist
        
    conn.commit()
    conn.close()

def log_analytics(question, answer, confidence, latency, escalated, docs, input_tokens, output_tokens, cost, model):
    """Saves telemetry to the database for the Streamlit Dashboard."""
    conn = sqlite3.connect('data/analytics.db')
    c = conn.cursor()
    docs_json = json.dumps(docs)
    c.execute('''INSERT INTO analytics (question, answer, confidence, latency, escalated, docs, date, input_tokens, output_tokens, total_cost, model_used) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
              (question, answer, confidence, latency, escalated, docs_json, str(datetime.datetime.now()), input_tokens, output_tokens, cost, model))
    row_id = c.lastrowid
    conn.commit()
    conn.close()
    return row_id

def update_feedback(log_id, is_helpful):
    """Updates the analytics row with human-in-the-loop feedback."""
    conn = sqlite3.connect('data/analytics.db')
    c = conn.cursor()
    c.execute('UPDATE analytics SET helpful = ? WHERE id = ?', (is_helpful, log_id))
    conn.commit()
    conn.close()

def stream_text(text):
    """Helper to simulate streaming UI, since full processing occurs in the backend."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

# Set up the basic look of the page
st.set_page_config(
    page_title="Eskalyr AI",
    layout="centered",
)

init_db()

# Uses cache_resource to avoid rebuilding the AI pipeline on every button click
@st.cache_resource
def get_chain():
    return build_chain()

# Initialize the chat history in the session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handles pending queries triggered via the sidebar buttons
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# --- Side Menu Setup ---
with st.sidebar:
    st.title("Eskalyr AI")
    st.caption("Powered by RAG · Cross-Encoder Reranking · Groq")
    st.divider()

    st.markdown("**Sample questions**")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    
    # Button to wipe the chat memory clean
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []

# --- Main Chat Area Setup ---
st.title("Eskalyr: Enterprise Support")
st.caption("Ask anything. Equipped with Confidence Scoring, Agentic Escalation, and Hallucination Detection.")

# Render all the past messages on the screen
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Add feedback buttons for Assistant messages
        if msg["role"] == "assistant" and "log_id" in msg and i == len(st.session_state.messages) - 1:
            col1, col2, _ = st.columns([1, 1, 8])
            with col1:
                if st.button("Helpful", key=f"up_{i}"):
                    update_feedback(msg["log_id"], True)
                    st.toast("Feedback saved. Thank you.")
            with col2:
                if st.button("Not Helpful", key=f"down_{i}"):
                    update_feedback(msg["log_id"], False)
                    st.toast("Feedback saved. Will improve.")

# Process input query (either typed or clicked)
question = st.chat_input("Describe the issue…")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

# Process the question if available
if question:
    # Save the user's question to the chat history
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Processing with Enterprise RAG..."):
            chain = get_chain()
            
            formatted_history = []
            for msg in st.session_state.messages[:-1]:
                role = "human" if msg["role"] == "user" else "assistant"
                formatted_history.append((role, msg["content"]))
                
            # Run the pipeline
            result = chain.invoke({"question": question, "chat_history": formatted_history})
            
            # Log all analytics to the database
            log_id = log_analytics(
                question=question,
                answer=result["answer"],
                confidence=result["confidence"],
                latency=result["latency"],
                escalated=result["escalated"],
                docs=result["docs"],
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                cost=result.get("cost", 0.0),
                model=result.get("model", "unknown")
            )
            
            # Display the answer in a streaming fashion
            st.write_stream(stream_text(result["answer"]))

    # Save the AI's final answer + the DB log_id to the chat history
    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "log_id": log_id})
    st.rerun() # Force a rerun so the feedback buttons appear
