"""
Evaluation script for testing the retrieval quality of the tickets store.
It measures top-k recall (k=3) for 10 hand-crafted test cases.
Usage: python evaluate_retrieval.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 10 hand-crafted question and expected ticket ID pairs
EVAL_DATASET = [
    {
        "question": "My calls and internet are dropping to zero multiple times a day in a suburban area",
        "expected_ticket_id": "TK-002"
    },
    {
        "question": "I bought a 5 GB data add-on but my balance hasn't updated in the app",
        "expected_ticket_id": "TK-003"
    },
    {
        "question": "After updating my Android phone, I got a 'SIM not provisioned' error",
        "expected_ticket_id": "TK-005"
    },
    {
        "question": "I see a double charge of 45 dollars on my bank statement on the same date",
        "expected_ticket_id": "TK-006"
    },
    {
        "question": "Incoming calls go straight to my voicemail even though I have full signal and DND is off",
        "expected_ticket_id": "TK-007"
    },
    {
        "question": "My 4G internet speed in the city center is super slow, under 1 Mbps",
        "expected_ticket_id": "TK-008"
    },
    {
        "question": "QR code scan is failing when I try to activate eSIM on my iPhone 15",
        "expected_ticket_id": "TK-009"
    },
    {
        "question": "I'm travelling in Tokyo, Japan, but I have absolutely no mobile signal",
        "expected_ticket_id": "TK-012"
    },
    {
        "question": "Tethering / hotspot is blocked on my Android device even though I'm on an unlimited plan",
        "expected_ticket_id": "TK-014"
    },
    {
        "question": "I hear my own voice echoing back with a 1-second delay when making outgoing calls",
        "expected_ticket_id": "TK-017"
    }
]

def main():
    print("==========================================================")
    print("Evaluating Retrieval Quality: Ticket Collection (Top-3 Recall)")
    print("==========================================================\n")

    print("Loading embedding model and ticket vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    hits = 0
    total = len(EVAL_DATASET)

    for idx, item in enumerate(EVAL_DATASET, 1):
        question = item["question"]
        expected_id = item["expected_ticket_id"]

        print(f"Test Case {idx}/{total}:")
        print(f"  Question   : '{question}'")
        print(f"  Expected ID: {expected_id}")

        # Retrieve top 3 documents from Chroma
        results = tickets_store.similarity_search_with_score(question, k=3)

        retrieved_ids = []
        retrieved_details = []
        for doc, score in results:
            t_id = doc.metadata.get("ticket_id", "unknown")
            retrieved_ids.append(t_id)
            # Cosine similarity = 1 - (distance^2 / 2)
            similarity = 1.0 - (score / 2.0)
            retrieved_details.append(f"{t_id} (sim: {similarity:.4f})")

        print(f"  Retrieved  : {', '.join(retrieved_details)}")

        # Check if expected ticket is in the top 3
        if expected_id in retrieved_ids:
            hits += 1
            print("  Result     : [HIT]")
        else:
            print("  Result     : [MISS]")
        print("-" * 58)

    recall = (hits / total) * 100
    print(f"\nFinal Metric: Top-3 Recall = {recall:.1f}% ({hits}/{total} hits)")
    print("==========================================================")

if __name__ == "__main__":
    main()
