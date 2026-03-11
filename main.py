import os
from dotenv import load_dotenv

from dataset_loader import load_squad_dataset
from retriever import VectorStore
from generator import generate_answer
from evaluator import evaluate_generation, evaluate_retrieval


def main():
    print("Program started...")
    load_dotenv()

    print("Loading dataset...")
    contexts, qa_data = load_squad_dataset("dev-v1.1.json")

    print("Building vector index...")
    vector_store = VectorStore(contexts)

    print("Running RAG pipeline on first 20 questions...\n")

    results = []
    retrieval_results = []

    k = 3  # top-k retrieval

    for item in qa_data[:20]:
        question = item["question"]
        actual_answer = item["answer"]
        correct_context = item["context"]

        print("Question:", question)

        # Retrieve indices
        retrieved_indices = vector_store.retrieve(
            question, top_k=k, return_indices=True
        )

        retrieved_contexts = [contexts[i] for i in retrieved_indices]

        # Generate answer
        generated_answer = generate_answer(retrieved_contexts, question)

        print("Generated:", generated_answer)
        print("Actual:", actual_answer)
        print("-" * 60)

        # Store generation results
        results.append((generated_answer, actual_answer))

        # Store retrieval results
        correct_index = contexts.index(correct_context)
        retrieval_results.append((retrieved_indices, correct_index))

    # Evaluate generation
    gen_metrics = evaluate_generation(results)

    # Evaluate retrieval
    ret_metrics = evaluate_retrieval(retrieval_results, k)

    print("\n========== FINAL EVALUATION ==========")

    print("\n--- Generation Metrics ---")
    print("Exact Match:", round(gen_metrics["Exact Match"], 3))
    print("F1 Score:", round(gen_metrics["F1 Score"], 3))

    print("\n--- Retrieval Metrics ---")
    for key, value in ret_metrics.items():
        print(f"{key}:", round(value, 3))


if __name__ == "__main__":
    main()