import sqlite3
import pandas as pd
import streamlit as st
import json

st.set_page_config(
    page_title="RAG Analytics Dashboard",
    layout="wide",
)

st.title("Eskalyr Analytics Dashboard")
st.markdown("Monitor Retrieval Confidence, Latency, Escalations, Cost, and User Feedback in real-time.")

def load_data():
    try:
        conn = sqlite3.connect('data/analytics.db')
        df = pd.read_sql_query("SELECT * FROM analytics", conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.info("No analytics data found yet. Start chatting with the Enterprise AI to generate logs.")
    st.stop()

# --- Top Level Metrics ---
total_queries = len(df)
avg_confidence = df['confidence'].mean()
avg_latency = df['latency'].mean()
escalation_rate = (df['escalated'].sum() / total_queries) * 100

# Cost Metrics (if available)
total_cost = df['total_cost'].sum() if 'total_cost' in df.columns else 0.0
avg_cost = df['total_cost'].mean() if 'total_cost' in df.columns else 0.0
total_input_tokens = df['input_tokens'].sum() if 'input_tokens' in df.columns else 0
total_output_tokens = df['output_tokens'].sum() if 'output_tokens' in df.columns else 0

# Feedback Metrics
feedback_df = df.dropna(subset=['helpful'])
if not feedback_df.empty:
    helpful_rate = (feedback_df['helpful'].sum() / len(feedback_df)) * 100
else:
    helpful_rate = 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Queries", total_queries)
col2.metric("Avg Confidence", f"{avg_confidence:.1f}%")
col3.metric("Avg Latency", f"{avg_latency:.2f}s")
col4.metric("Escalation Rate", f"{escalation_rate:.1f}%")
col5.metric("Helpful Rating", f"{helpful_rate:.1f}%")

st.divider()

# --- Cost & Token Tracking ---
st.subheader("Cost & Usage Tracking")
cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)
cost_col1.metric("Total Spend", f"${total_cost:.4f}")
cost_col2.metric("Average Cost / Query", f"${avg_cost:.4f}")
cost_col3.metric("Total Input Tokens", f"{total_input_tokens:,}")
cost_col4.metric("Total Output Tokens", f"{total_output_tokens:,}")

st.divider()

# --- Charts ---
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Confidence Trends")
    st.line_chart(df.set_index('id')['confidence'])

with col_chart2:
    st.subheader("Query Escalations")
    escalation_counts = df['escalated'].value_counts().rename({0: "Answered", 1: "Escalated"})
    st.bar_chart(escalation_counts)

st.divider()

# --- Continuous Improvement Pipeline (Failed Queries) ---
st.subheader("Continuous Improvement Pipeline")
st.markdown("Review queries that received a 'Not Helpful' to evaluate the retrieved documents and improve the knowledge base.")

failed_queries = df[df['helpful'] == 0]

if failed_queries.empty:
    st.success("No failed queries logged yet.")
else:
    for idx, row in failed_queries.iterrows():
        with st.expander(f"Question: {row['question']} (Confidence: {row['confidence']:.1f}%)"):
            st.markdown("**AI Answer:**")
            st.info(row['answer'])
            
            st.markdown("**Retrieved Sources:**")
            try:
                docs = json.loads(row['docs'])
                if not docs:
                    st.warning("No documents were retrieved for this query.")
                for d in docs:
                    st.json(d)
            except Exception:
                st.write("Could not parse retrieved docs.")

st.divider()

# --- Raw Telemetry Data ---
st.subheader("Raw Telemetry Logs")
display_df = df.copy()
display_df = display_df.sort_values(by='id', ascending=False)
st.dataframe(display_df, use_container_width=True)
