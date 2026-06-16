"""Retrieval and lightweight RAG quality metrics for Day 14."""

from __future__ import annotations

import re
from typing import Dict, List


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "")}


def _overlap_score(a: str, b: str) -> float:
    left = _tokens(a)
    right = _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


class RetrievalEvaluator:
    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        for index, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (index + 1)
        return 0.0

    def calculate_recall(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 5) -> float:
        if not expected_ids:
            return 1.0
        found = set(expected_ids) & set(retrieved_ids[:top_k])
        return len(found) / len(set(expected_ids))

    def calculate_precision(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 5) -> float:
        top_retrieved = retrieved_ids[:top_k]
        if not top_retrieved:
            return 1.0 if not expected_ids else 0.0
        found = set(expected_ids) & set(top_retrieved)
        return len(found) / len(top_retrieved)

    async def score(self, test_case: Dict, response: Dict) -> Dict:
        expected_doc_ids = test_case.get("ground_truth_doc_ids") or test_case.get("expected_retrieval_ids", [])
        expected_chunk_ids = test_case.get("ground_truth_chunk_ids", [])
        retrieved_doc_ids = response.get("retrieved_ids", [])
        retrieved_chunk_ids = response.get("retrieved_chunk_ids", [])
        contexts = response.get("contexts", [])
        answer = response.get("answer", "")

        doc_hit = self.calculate_hit_rate(expected_doc_ids, retrieved_doc_ids, top_k=3)
        chunk_hit = self.calculate_hit_rate(expected_chunk_ids, retrieved_chunk_ids, top_k=3)
        hit_rate = max(doc_hit, chunk_hit)
        mrr = max(
            self.calculate_mrr(expected_doc_ids, retrieved_doc_ids),
            self.calculate_mrr(expected_chunk_ids, retrieved_chunk_ids),
        )
        recall = max(
            self.calculate_recall(expected_doc_ids, retrieved_doc_ids, top_k=5),
            self.calculate_recall(expected_chunk_ids, retrieved_chunk_ids, top_k=5),
        )
        precision = max(
            self.calculate_precision(expected_doc_ids, retrieved_doc_ids, top_k=5),
            self.calculate_precision(expected_chunk_ids, retrieved_chunk_ids, top_k=5),
        )

        context_text = " ".join(contexts)
        faithfulness = _overlap_score(answer, context_text)
        relevancy = _overlap_score(test_case.get("expected_answer", ""), answer)
        context_precision = precision
        context_recall = recall

        return {
            "faithfulness": round(faithfulness, 4),
            "relevancy": round(relevancy, 4),
            "answer_relevance": round(relevancy, 4),
            "context_precision": round(context_precision, 4),
            "context_recall": round(context_recall, 4),
            "retrieval": {
                "hit_rate": round(hit_rate, 4),
                "mrr": round(mrr, 4),
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "retrieved_ids": retrieved_doc_ids,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "expected_ids": expected_doc_ids,
                "expected_chunk_ids": expected_chunk_ids,
            },
        }

    async def evaluate_batch(self, dataset: List[Dict], responses: List[Dict]) -> Dict:
        scores = [await self.score(case, response) for case, response in zip(dataset, responses)]
        total = max(len(scores), 1)
        return {
            "avg_hit_rate": sum(item["retrieval"]["hit_rate"] for item in scores) / total,
            "avg_mrr": sum(item["retrieval"]["mrr"] for item in scores) / total,
            "avg_faithfulness": sum(item["faithfulness"] for item in scores) / total,
            "avg_relevancy": sum(item["relevancy"] for item in scores) / total,
        }
