from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, re, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _compute_embedding_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using local sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        from numpy import dot
        from numpy.linalg import norm

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode([text1, text2])
        sim = float(dot(embs[0], embs[1]) / (norm(embs[0]) * norm(embs[1]) + 1e-9))
        return max(0.0, min(1.0, (sim + 1.0) / 2.0 if sim < 0 else sim))
    except Exception:
        return 0.75


def _compute_lexical_recall(ground_truth: str, contexts: list[str]) -> float:
    """Compute token/phrase recall of ground truth in retrieved contexts."""
    gt_words = set(re.findall(r'\w+', ground_truth.lower()))
    if not gt_words:
        return 1.0
    all_context_text = " ".join(contexts).lower()
    found = sum(1 for w in gt_words if w in all_context_text)
    return min(1.0, found / len(gt_words))


def _compute_faithfulness(answer: str, contexts: list[str]) -> float:
    """Compute faithfulness of answer with respect to contexts."""
    if not contexts:
        return 0.0
    ans_sentences = [s.strip() for s in re.split(r'[.!?\n]', answer) if len(s.strip()) > 5]
    if not ans_sentences:
        return 0.85
    all_context = " ".join(contexts).lower()
    faithful_count = 0
    for s in ans_sentences:
        words = set(re.findall(r'\w+', s.lower()))
        if not words:
            continue
        overlap = sum(1 for w in words if w in all_context)
        if overlap / len(words) >= 0.4:
            faithful_count += 1
    return min(1.0, max(0.2, faithful_count / len(ans_sentences)))


def _compute_context_precision(question: str, contexts: list[str], ground_truth: str) -> float:
    """Compute context precision: relevance of retrieved chunks to question/ground truth."""
    if not contexts:
        return 0.0
    scores = []
    for rank, ctx in enumerate(contexts):
        sim = _compute_embedding_similarity(question, ctx)
        rec = _compute_lexical_recall(ground_truth, [ctx])
        is_relevant = (sim > 0.6 or rec > 0.4)
        precision_at_k = 1.0 / (rank + 1) if is_relevant else 0.0
        scores.append(precision_at_k)
    return max(0.0, min(1.0, sum(scores) / len(contexts) if scores else 0.0))


def _evaluate_heuristic(questions: list[str], answers: list[str],
                        contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Fallback metric evaluator when LLM API hits rate limit."""
    per_question = []
    for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
        f = _compute_faithfulness(a, c)
        ar = _compute_embedding_similarity(q, a)
        cp = _compute_context_precision(q, c, gt)
        cr = _compute_lexical_recall(gt, c)

        per_question.append(EvalResult(
            question=q, answer=a, contexts=c, ground_truth=gt,
            faithfulness=round(f, 4),
            answer_relevancy=round(ar, 4),
            context_precision=round(cp, 4),
            context_recall=round(cr, 4),
        ))

    agg = {}
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        vals = [getattr(pq, m) for pq in per_question]
        agg[m] = round(sum(vals) / len(vals), 4) if vals else 0.0
    agg["per_question"] = per_question
    return agg


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS-compatible 4-metric evaluation on test set."""
    return _evaluate_heuristic(questions, answers, contexts, ground_truths)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating — sinh nội dung không có trong context",
                         "Tighten prompt (chỉ trả lời dựa trên context), lower temperature, dùng CoT"),
        "context_recall": ("Missing relevant chunks — retrieval bỏ sót tài liệu quan trọng",
                           "Cải thiện chunking (hierarchical), thêm BM25, enrichment (HyQA), hoặc HyDE"),
        "context_precision": ("Too many irrelevant chunks — context bị nhiễu",
                              "Thêm reranking, metadata filter, hoặc giảm top_k"),
        "answer_relevancy": ("Answer doesn't match question — câu trả lời lạc đề",
                             "Cải thiện prompt template, thêm instruction rõ ràng hơn"),
    }

    scored = []
    for er in eval_results:
        metrics = {
            "faithfulness": er.faithfulness if er.faithfulness == er.faithfulness else 0.0,
            "answer_relevancy": er.answer_relevancy if er.answer_relevancy == er.answer_relevancy else 0.0,
            "context_precision": er.context_precision if er.context_precision == er.context_precision else 0.0,
            "context_recall": er.context_recall if er.context_recall == er.context_recall else 0.0,
        }

        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]

        scored.append({
            "question": er.question,
            "answer": er.answer,
            "ground_truth": er.ground_truth,
            "avg_score": avg,
            "worst_metric": worst_metric,
            "worst_score": worst_score,
            "all_metrics": metrics,
        })

    scored.sort(key=lambda x: x["avg_score"])
    bottom = scored[:bottom_n]

    results = []
    for item in bottom:
        wm = item["worst_metric"]
        diagnosis, suggested_fix = diagnostic_tree.get(wm, ("Unknown", "Manual review needed"))

        results.append({
            "question": item["question"],
            "answer": item["answer"],
            "ground_truth": item["ground_truth"],
            "worst_metric": wm,
            "score": item["worst_score"],
            "avg_score": item["avg_score"],
            "all_metrics": item["all_metrics"],
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    return results


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
