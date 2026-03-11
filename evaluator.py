import re
from collections import Counter


def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = " ".join(s.split())
    return s


def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_generation(results):
    exact_scores = []
    f1_scores = []

    for pred, actual in results:
        exact_scores.append(exact_match(pred, actual))
        f1_scores.append(f1_score(pred, actual))

    return {
        "Exact Match": sum(exact_scores) / len(exact_scores),
        "F1 Score": sum(f1_scores) / len(f1_scores),
    }


def evaluate_retrieval(retrieval_results, k):
    recall = 0
    precision = 0
    mrr = 0

    for retrieved_indices, correct_index in retrieval_results:
        if correct_index in retrieved_indices:
            recall += 1
            precision += 1 / k
            rank = list(retrieved_indices).index(correct_index) + 1
            mrr += 1 / rank

    total = len(retrieval_results)

    return {
        f"Recall@{k}": recall / total,
        f"Precision@{k}": precision / total,
        "MRR": mrr / total,
    }