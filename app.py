import streamlit as st
from dotenv import load_dotenv
import time

from dataset_loader import load_squad_dataset
from retriever import VectorStore
from generator import generate_answer
from evaluator import evaluate_generation, evaluate_retrieval

load_dotenv()

st.set_page_config(page_title="RAG System", layout="wide")

st.title("📚 Retrieval-Augmented Generation (RAG) System")

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider("Top-K Retrieval", 1, 10, 3)
eval_size = st.sidebar.slider("Evaluation Questions", 5, 50, 20)

# -------------------------
# Load System (Cached)
# -------------------------
@st.cache_resource
def load_system():
    contexts, qa_data = load_squad_dataset("dev-v1.1.json")
    vector_store = VectorStore(contexts)
    return contexts, qa_data, vector_store

contexts, qa_data, vector_store = load_system()

# =========================
# 🔎 QUERY MODE
# =========================
st.header("🔎 Ask a Question")

user_question = st.text_input("Enter your question:")

if user_question:

    start_time = time.time()

    retrieved_indices = vector_store.retrieve(
        user_question, top_k=top_k, return_indices=True
    )
    retrieved_contexts = [contexts[i] for i in retrieved_indices]

    answer = generate_answer(retrieved_contexts, user_question)

    latency = round(time.time() - start_time, 2)

    st.subheader("🤖 Generated Answer")
    st.write(answer)

    st.caption(f"⏱ Response Time: {latency} seconds")

    st.subheader("📄 Retrieved Contexts")
    for i, ctx in enumerate(retrieved_contexts):
        with st.expander(f"Rank {i+1}"):
            st.write(ctx)

# =========================
# 📊 EVALUATION MODE
# =========================
st.header("📊 Run Evaluation")

if st.button("Run Evaluation"):

    st.write("Running evaluation...")

    generation_results = []
    retrieval_results = []

    progress_bar = st.progress(0)

    for idx, item in enumerate(qa_data[:eval_size]):

        question = item["question"]
        actual_answer = item["answer"]
        correct_context = item["context"]

        retrieved_indices = vector_store.retrieve(
            question, top_k=top_k, return_indices=True
        )
        retrieved_contexts = [contexts[i] for i in retrieved_indices]

        generated_answer = generate_answer(retrieved_contexts, question)

        generation_results.append((generated_answer, actual_answer))

        correct_index = contexts.index(correct_context)
        retrieval_results.append((retrieved_indices, correct_index))

        progress_bar.progress((idx + 1) / eval_size)

    gen_metrics = evaluate_generation(generation_results)
    ret_metrics = evaluate_retrieval(retrieval_results, top_k)

    st.success("Evaluation Completed ✅")

    # -------------------------
    # Generation Metrics
    # -------------------------
    st.subheader("🧠 Generation Metrics")

    col1, col2 = st.columns(2)
    col1.metric("Exact Match", round(gen_metrics["Exact Match"], 3))
    col2.metric("F1 Score", round(gen_metrics["F1 Score"], 3))

    # -------------------------
    # Retrieval Metrics
    # -------------------------
    st.subheader("📡 Retrieval Metrics")

    col3, col4, col5 = st.columns(3)

    col3.metric(f"Recall@{top_k}", round(ret_metrics[f"Recall@{top_k}"], 3))
    col4.metric(f"Precision@{top_k}", round(ret_metrics[f"Precision@{top_k}"], 3))
    col5.metric("MRR", round(ret_metrics["MRR"], 3))