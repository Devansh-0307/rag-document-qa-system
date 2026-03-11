AI-Powered Document Question Answering System using RAG :
This project implements a Retrieval-Augmented Generation (RAG) pipeline for answering questions using external documents.
The system retrieves relevant contexts from a document dataset and uses a Large Language Model (LLM) to generate accurate answers grounded in those contexts.
The project also evaluates both retrieval quality and generation quality using multiple metrics.

Features :
 > Document Question Answering using Retrieval-Augmented Generation (RAG)
 > Semantic search using SentenceTransformers embeddings
 > Efficient vector retrieval using FAISS
 > Answer generation using Groq LLaMA model
 > Evaluation of system performance with:
      > Exact Match (EM)
      > F1 Score
      > Recall@K
      > Precision@K
      > Mean Reciprocal Rank (MRR)
      > Interactive UI built with Streamlit

System Architecture :
User Question
      ↓
Embedding Model (SentenceTransformers)
      ↓
Vector Search (FAISS)
      ↓
Top-K Retrieved Contexts
      ↓
LLM (Groq - LLaMA 3.1)
      ↓
Generated Answer
      ↓
Evaluation Metrics

Technologies Used
> Python
> SentenceTransformers
> FAISS
> Groq API
> Streamlit
> NumPy
> SQuAD Dataset

Evaluation Metrics
The system evaluates both retrieval and generation performance.
Generation Metrics :
  > Exact Match (EM) – checks if predicted answer exactly matches the ground truth.
  > F1 Score – measures token overlap between predicted and actual answers.
Retrieval Metrics :
  > Recall@K – probability that the correct document appears in top K results.
  > Precision@K – ratio of relevant documents among retrieved results.
  > MRR (Mean Reciprocal Rank) – evaluates ranking quality of the correct document.
