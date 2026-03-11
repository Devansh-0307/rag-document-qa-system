import json


def load_squad_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    contexts = []
    qa_pairs = []

    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            contexts.append(context)

            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = qa["answers"][0]["text"]

                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "context": context
                })

    return contexts, qa_pairs 