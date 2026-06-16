"""Async benchmark runner for Day 14."""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List


def classify_failure(test_case: Dict, response: Dict, ragas_scores: Dict, judge_result: Dict) -> str:
    if judge_result.get("final_score", 0) >= 3 and ragas_scores.get("retrieval", {}).get("hit_rate", 0) >= 0.5:
        return "none"
    if not test_case.get("ground_truth_doc_ids") and response.get("retrieved_ids"):
        return "question_out_of_scope"
    if ragas_scores.get("retrieval", {}).get("hit_rate", 0) == 0:
        return "retrieval_miss"
    if ragas_scores.get("faithfulness", 0) < 0.35:
        return "generation_hallucination"
    if "[" not in response.get("answer", ""):
        return "citation_missing_or_wrong"
    return "answer_incomplete"


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge, concurrency: int = 8):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.semaphore = asyncio.Semaphore(concurrency)

    async def run_single_test(self, test_case: Dict) -> Dict:
        async with self.semaphore:
            start_time = time.perf_counter()
            try:
                response = await self.agent.query(test_case["question"])
                latency = time.perf_counter() - start_time
                ragas_scores = await self.evaluator.score(test_case, response)
                judge_result = await self.judge.evaluate_multi_judge(
                    test_case["question"],
                    response["answer"],
                    test_case["expected_answer"],
                )
                failure_stage = classify_failure(test_case, response, ragas_scores, judge_result)
                return {
                    "id": test_case.get("id"),
                    "test_case": test_case["question"],
                    "category": test_case.get("category"),
                    "difficulty": test_case.get("difficulty"),
                    "expected_answer": test_case.get("expected_answer"),
                    "expected_retrieval_ids": test_case.get("ground_truth_doc_ids", []),
                    "agent_response": response["answer"],
                    "retrieved_ids": response.get("retrieved_ids", []),
                    "retrieved_chunk_ids": response.get("retrieved_chunk_ids", []),
                    "latency": round(latency, 4),
                    "token_usage": response.get("metadata", {}).get("tokens_used", 0),
                    "estimated_cost_usd": response.get("metadata", {}).get("estimated_cost_usd", 0.0),
                    "ragas": ragas_scores,
                    "judge": judge_result,
                    "failure_stage": failure_stage,
                    "status": "pass" if judge_result["final_score"] >= 3 else "fail",
                }
            except Exception as exc:
                latency = time.perf_counter() - start_time
                return {
                    "id": test_case.get("id"),
                    "test_case": test_case.get("question"),
                    "latency": round(latency, 4),
                    "status": "error",
                    "failure_stage": "runner_error",
                    "error": str(exc),
                    "ragas": {
                        "faithfulness": 0.0,
                        "relevancy": 0.0,
                        "answer_relevance": 0.0,
                        "context_precision": 0.0,
                        "context_recall": 0.0,
                        "retrieval": {"hit_rate": 0.0, "mrr": 0.0, "recall": 0.0, "precision": 0.0},
                    },
                    "judge": {"final_score": 1.0, "agreement_rate": 0.0},
                }

    async def run_all(self, dataset: List[Dict]) -> List[Dict]:
        tasks = [self.run_single_test(case) for case in dataset]
        return await asyncio.gather(*tasks)
