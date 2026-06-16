"""Validate Day 14 submission artifacts.

The output intentionally uses ASCII so it works on Windows terminals that do
not default to UTF-8.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def validate_lab() -> bool:
    print("[CHECK] Validating Day 14 submission format...")

    required_files = [
        ROOT / "data" / "golden_set.jsonl",
        ROOT / "reports" / "summary.json",
        ROOT / "reports" / "benchmark_results.json",
        ROOT / "analysis" / "failure_analysis.md",
    ]

    missing = []
    for path in required_files:
        if path.exists():
            print(f"[OK] Found: {path.relative_to(ROOT)}")
        else:
            print(f"[FAIL] Missing: {path.relative_to(ROOT)}")
            missing.append(path)

    if missing:
        print(f"[FAIL] Missing {len(missing)} required files.")
        return False

    total_cases = count_jsonl(ROOT / "data" / "golden_set.jsonl")
    if total_cases < 50:
        print(f"[FAIL] Golden dataset has {total_cases} cases; expected at least 50.")
        return False
    print(f"[OK] Golden dataset cases: {total_cases}")

    summary = load_json(ROOT / "reports" / "summary.json")
    benchmark = load_json(ROOT / "reports" / "benchmark_results.json")

    if "metrics" not in summary or "metadata" not in summary:
        print("[FAIL] summary.json must contain 'metrics' and 'metadata'.")
        return False

    metrics = summary["metrics"]
    required_metrics = [
        "avg_score",
        "hit_rate",
        "mrr",
        "agreement_rate",
        "avg_latency_ms",
        "estimated_cost_usd",
    ]
    for metric in required_metrics:
        if metric not in metrics:
            print(f"[FAIL] Missing metric: {metric}")
            return False
        print(f"[OK] Metric {metric}: {metrics[metric]}")

    if "gate" not in summary or summary["gate"].get("decision") not in {"release", "rollback"}:
        print("[FAIL] summary.json must include gate.decision as release/rollback.")
        return False
    print(f"[OK] Release gate: {summary['gate']['decision']}")

    candidate_results = benchmark.get("candidate", [])
    if len(candidate_results) < 50:
        print(f"[FAIL] benchmark_results.json has {len(candidate_results)} candidate cases; expected at least 50.")
        return False
    print(f"[OK] Candidate benchmark cases: {len(candidate_results)}")

    print("[PASS] Lab is ready for submission.")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if validate_lab() else 1)
