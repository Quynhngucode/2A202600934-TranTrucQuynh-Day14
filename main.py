"""Run the Day 14 AI Evaluation Factory benchmark."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

from agent.main_agent import MainAgent
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "data" / "golden_set.jsonl"
REPORTS_DIR = ROOT / "reports"
ANALYSIS_PATH = ROOT / "analysis" / "failure_analysis.md"


def load_dataset() -> List[Dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError("Missing data/golden_set.jsonl. Run: python data/synthetic_gen.py")
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    if len(dataset) < 50:
        raise ValueError(f"Golden dataset must contain at least 50 cases; found {len(dataset)}.")
    return dataset


def average(results: List[Dict], getter, default: float = 0.0) -> float:
    values = []
    for result in results:
        try:
            values.append(float(getter(result)))
        except (KeyError, TypeError, ValueError):
            values.append(default)
    return sum(values) / max(len(values), 1)


def summarize_results(agent_version: str, results: List[Dict]) -> Dict:
    total = len(results)
    pass_count = sum(1 for result in results if result.get("status") == "pass")
    latencies = [float(result.get("latency", 0.0)) for result in results]
    failure_counts = Counter(result.get("failure_stage", "unknown") for result in results)
    return {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "pass_rate": round(pass_count / max(total, 1), 4),
            "avg_score": round(average(results, lambda r: r["judge"]["final_score"]), 4),
            "hit_rate": round(average(results, lambda r: r["ragas"]["retrieval"]["hit_rate"]), 4),
            "mrr": round(average(results, lambda r: r["ragas"]["retrieval"]["mrr"]), 4),
            "context_recall": round(average(results, lambda r: r["ragas"]["context_recall"]), 4),
            "context_precision": round(average(results, lambda r: r["ragas"]["context_precision"]), 4),
            "faithfulness": round(average(results, lambda r: r["ragas"]["faithfulness"]), 4),
            "answer_relevance": round(average(results, lambda r: r["ragas"]["answer_relevance"]), 4),
            "agreement_rate": round(average(results, lambda r: r["judge"]["agreement_rate"]), 4),
            "avg_latency_ms": round(1000 * sum(latencies) / max(total, 1), 2),
            "p95_latency_ms": round(1000 * statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) >= 20 else round(1000 * max(latencies, default=0), 2),
            "total_tokens": int(sum(result.get("token_usage", 0) for result in results)),
            "estimated_cost_usd": round(sum(float(result.get("estimated_cost_usd", 0.0)) for result in results), 6),
        },
        "failure_counts": dict(failure_counts),
    }


def release_gate(base: Dict, candidate: Dict) -> Dict:
    base_metrics = base["metrics"]
    candidate_metrics = candidate["metrics"]
    reasons = []

    score_delta = candidate_metrics["avg_score"] - base_metrics["avg_score"]
    hit_delta = candidate_metrics["hit_rate"] - base_metrics["hit_rate"]
    mrr_delta = candidate_metrics["mrr"] - base_metrics["mrr"]
    cost_delta = candidate_metrics["estimated_cost_usd"] - base_metrics["estimated_cost_usd"]
    latency_delta = candidate_metrics["avg_latency_ms"] - base_metrics["avg_latency_ms"]

    if score_delta < -0.05:
        reasons.append(f"Average judge score regressed by {score_delta:.3f}.")
    if hit_delta < -0.05:
        reasons.append(f"Hit Rate regressed by {hit_delta:.3f}.")
    if mrr_delta < -0.05:
        reasons.append(f"MRR regressed by {mrr_delta:.3f}.")
    if cost_delta > 0.001 and score_delta < 0.1:
        reasons.append("Cost increased without meaningful quality gain.")
    if candidate_metrics["avg_latency_ms"] > 2000:
        reasons.append("Average latency exceeds 2s target.")
    if candidate_metrics["agreement_rate"] < 0.6:
        reasons.append("Judge agreement rate is below reliability threshold.")

    decision = "release" if not reasons else "rollback"
    if decision == "release":
        reasons.append("Candidate passes quality, retrieval, cost and latency gates.")

    return {
        "decision": decision,
        "reasons": reasons,
        "delta": {
            "avg_score": round(score_delta, 4),
            "hit_rate": round(hit_delta, 4),
            "mrr": round(mrr_delta, 4),
            "estimated_cost_usd": round(cost_delta, 6),
            "avg_latency_ms": round(latency_delta, 2),
        },
    }


async def run_benchmark(agent_version: str, dataset: List[Dict]) -> tuple[List[Dict], Dict]:
    runner = BenchmarkRunner(
        MainAgent(version=agent_version),
        RetrievalEvaluator(),
        LLMJudge(),
        concurrency=10,
    )
    results = await runner.run_all(dataset)
    summary = summarize_results(agent_version, results)
    return results, summary


def export_failure_analysis(results: List[Dict], summary: Dict) -> None:
    worst = sorted(results, key=lambda item: item.get("judge", {}).get("final_score", 0))[:5]
    lines = [
        "# Bao cao Phan tich That bai (Failure Analysis Report)",
        "",
        "## 1. Tong quan Benchmark",
        f"- Tong so cases: {summary['metadata']['total']}",
        f"- Pass rate: {summary['metrics']['pass_rate']:.2%}",
        f"- Avg judge score: {summary['metrics']['avg_score']:.2f} / 5.0",
        f"- Hit Rate: {summary['metrics']['hit_rate']:.2%}",
        f"- MRR: {summary['metrics']['mrr']:.3f}",
        f"- Agreement Rate: {summary['metrics']['agreement_rate']:.2%}",
        "",
        "## 2. Failure Clustering",
        "",
        "| Failure stage | Count | Root cause gia dinh |",
        "|---|---:|---|",
    ]
    for stage, count in sorted(summary.get("failure_counts", {}).items()):
        root = {
            "none": "Khong co loi nghiem trong.",
            "retrieval_miss": "Retriever khong dua ground-truth source vao top-k.",
            "generation_hallucination": "Answer co token khong duoc support boi context.",
            "citation_missing_or_wrong": "Prompt/formatter chua bat buoc citation chat.",
            "answer_incomplete": "Generation trich xuat thieu fact quan trong.",
            "question_out_of_scope": "Agent chua abstain tot voi cau hoi ngoai pham vi.",
        }.get(stage, "Can dieu tra them.")
        lines.append(f"| {stage} | {count} | {root} |")

    lines.extend(["", "## 3. Phan tich 5 Whys cho worst cases"])
    for index, item in enumerate(worst, start=1):
        lines.extend(
            [
                "",
                f"### Case #{index}: {item.get('id')} - {item.get('failure_stage')}",
                f"- Question: {item.get('test_case')}",
                f"- Score: {item.get('judge', {}).get('final_score')}",
                "1. Symptom: Case co diem thap hoac bi gan failure stage.",
                "2. Why 1: Chat luong retrieval/generation chua du manh cho dang cau hoi nay.",
                "3. Why 2: Cac token quan trong giua question va context khong khop hoan toan.",
                "4. Why 3: Golden case co the can them alias/tu dong nghia hoac reranking tot hon.",
                "5. Why 4: Pipeline hien tai la extractive offline, chua co LLM reasoning that.",
                "6. Root Cause: Can nang cap chunk metadata, synonym matching va judge LLM that neu co API key.",
            ]
        )

    lines.extend(
        [
            "",
            "## 4. Action Plan",
            "- [ ] Bo sung synonym dictionary tieng Viet khong dau/co dau cho retrieval.",
            "- [ ] Thu nghiem cross-encoder reranker cho cac case medium/hard.",
            "- [ ] Khi co API key, thay offline judge bang GPT/Claude va giu consensus layer hien co.",
            "- [ ] Mo rong hard cases thanh bo red-team rieng cho prompt injection va missing evidence.",
        ]
    )
    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} golden cases")

    base_results, base_summary = await run_benchmark("v1_base", dataset)
    candidate_results, candidate_summary = await run_benchmark("v2_hybrid_rerank", dataset)
    gate = release_gate(base_summary, candidate_summary)
    candidate_summary["baseline"] = base_summary
    candidate_summary["gate"] = gate

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "summary.json").write_text(
        json.dumps(candidate_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (REPORTS_DIR / "benchmark_results.json").write_text(
        json.dumps(
            {
                "baseline": base_results,
                "candidate": candidate_results,
                "comparison": {
                    "baseline_summary": base_summary,
                    "candidate_summary": candidate_summary,
                    "gate": gate,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    export_failure_analysis(candidate_results, candidate_summary)

    print("\n--- REGRESSION SUMMARY ---")
    print(f"V1 avg score: {base_summary['metrics']['avg_score']}")
    print(f"V2 avg score: {candidate_summary['metrics']['avg_score']}")
    print(f"Hit Rate: {candidate_summary['metrics']['hit_rate']}")
    print(f"MRR: {candidate_summary['metrics']['mrr']}")
    print(f"Agreement Rate: {candidate_summary['metrics']['agreement_rate']}")
    print(f"Gate decision: {gate['decision'].upper()}")
    print("Reports written to reports/summary.json and reports/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
