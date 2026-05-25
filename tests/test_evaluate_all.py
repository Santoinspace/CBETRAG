"""Task 8: Tests for evaluation and result aggregation (analysis/evaluate_all.py)."""
import json
import sys
import os
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.evaluate_all import _em, _f1, _estimate_lm_calls, aggregate, _find_results


# ── _em ───────────────────────────────────────────────────────────────────────

def test_em_exact_match():
    assert _em("hello world", "hello world") == 1.0


def test_em_case_insensitive():
    assert _em("Hello World", "hello world") == 1.0


def test_em_no_match():
    assert _em("hello", "world") == 0.0


def test_em_whitespace_insensitive():
    assert _em("  hello  ", "hello") == 1.0


# ── _f1 ───────────────────────────────────────────────────────────────────────

def test_f1_perfect_match():
    assert _f1("hello world", "hello world") == 1.0


def test_f1_no_match():
    assert _f1("hello", "world") == 0.0


def test_f1_partial_match():
    score = _f1("hello world foo", "hello world bar")
    # 3 pred tokens, 3 gold tokens; common: {hello, world} → prec=2/3, rec=2/3 → f1=2/3
    assert abs(score - 2 / 3) < 0.01


def test_f1_empty_pred():
    assert _f1("", "hello") == 0.0


def test_f1_empty_gold():
    assert _f1("hello", "") == 0.0


def test_f1_both_empty():
    assert _f1("", "") == 0.0


# ── _estimate_lm_calls ────────────────────────────────────────────────────────

def test_estimate_lm_calls_single_iteration():
    log = {"dag_size": 4, "iterations": 1}
    # 1 (DAG) + 1 × (4 × 3) + 1 (final) = 14
    assert _estimate_lm_calls(log) == 14


def test_estimate_lm_calls_multi_iteration():
    log = {"dag_size": 3, "iterations": 3}
    # 1 + 3 × (3 × 3) + 1 = 29
    assert _estimate_lm_calls(log) == 29


def test_estimate_lm_calls_defaults():
    log = {}
    # dag_size=1, iterations=1 → 1 + 1×3 + 1 = 5
    assert _estimate_lm_calls(log) == 5


# ── aggregate ─────────────────────────────────────────────────────────────────

SAMPLE_LOGS = [
    {
        "qid": "q1", "iterations": 2, "dag_size": 3,
        "branch_cs_scores": [0.8, 0.9, 0.7],
        "final_cs": 0.8,
        "conflicts_detected": ["q2"],
        "overrides_triggered": ["q2"],
        "noisy_evicted": [],
        "answer": "Paris",
        "gold_answer": "Paris",
        "em": 1, "f1": 1.0,
    },
    {
        "qid": "q2", "iterations": 3, "dag_size": 4,
        "branch_cs_scores": [0.6, 0.8, 0.9, 0.7],
        "final_cs": 0.7,
        "conflicts_detected": [],
        "overrides_triggered": [],
        "noisy_evicted": ["q3"],
        "answer": "London",
        "gold_answer": "London",
        "em": 1, "f1": 1.0,
    },
    {
        "qid": "q3", "iterations": 1, "dag_size": 2,
        "branch_cs_scores": [0.5, 0.6],
        "final_cs": 0.5,
        "conflicts_detected": ["q1"],
        "overrides_triggered": [],
        "noisy_evicted": [],
        "answer": "Berlin wall fell in 1990",
        "gold_answer": "Berlin wall fell in 1989",
        "em": 0, "f1": 0.0,
    },
]


def test_aggregate_basic():
    metrics = aggregate(SAMPLE_LOGS)
    assert metrics["n"] == 3
    # EM: (1+1+0)/3 = 2/3
    assert abs(metrics["em"] - 200 / 3) < 0.1
    # F1: (1.0+1.0+0.0)/3 = 2/3 * 100
    assert abs(metrics["f1"] - 200 / 3) < 0.1
    # avg retrieval rounds: (2+3+1)/3 = 2.0
    assert abs(metrics["avg_retrieval_rounds"] - 2.0) < 0.01
    # avg cs at stop: (0.8+0.7+0.5)/3 ≈ 0.667
    assert abs(metrics["avg_cs_at_stop"] - 2.0 / 3) < 0.1
    # conflict detected rate: 2/3 = 66.7%
    assert abs(metrics["conflict_detected_rate"] - 200 / 3) < 0.1
    # override triggered rate: 1/3 = 33.3%
    assert abs(metrics["override_triggered_rate"] - 100 / 3) < 0.1
    # noisy evicted rate: 1/3 = 33.3%
    assert abs(metrics["noisy_branch_evicted_rate"] - 100 / 3) < 0.1


def test_aggregate_empty():
    assert aggregate([]) == {}


def test_aggregate_single_entry():
    metrics = aggregate([SAMPLE_LOGS[0]])
    assert metrics["n"] == 1
    assert metrics["em"] == 100.0
    assert metrics["f1"] == 100.0
    assert metrics["avg_retrieval_rounds"] == 2.0


def test_aggregate_fallback_em_f1():
    """When 'em'/'f1' keys missing, compute from answer + gold_answer."""
    log_no_scores = {
        "qid": "test", "iterations": 1, "dag_size": 2,
        "answer": "  Hello World  ",
        "gold_answer": "hello world",
    }
    metrics = aggregate([log_no_scores])
    assert metrics["em"] == 100.0
    assert metrics["f1"] == 100.0


def test_aggregate_fallback_f1_mismatch():
    log = {
        "qid": "test", "iterations": 1, "dag_size": 2,
        "answer": "foo",
        "gold_answer": "bar",
    }
    metrics = aggregate([log])
    assert metrics["em"] == 0.0
    assert metrics["f1"] == 0.0


# ── _find_results integration ─────────────────────────────────────────────────

def test_find_results_with_synthetic_data():
    """Create temp JSON files and verify _find_results loads and aggregates them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cbet_data = [
            {"qid": "a", "iterations": 2, "dag_size": 2, "final_cs": 0.8,
             "answer": "X", "gold_answer": "X", "em": 1, "f1": 1.0,
             "conflicts_detected": [], "overrides_triggered": [], "noisy_evicted": []},
            {"qid": "b", "iterations": 3, "dag_size": 3, "final_cs": 0.6,
             "answer": "Y", "gold_answer": "Z", "em": 0, "f1": 0.0,
             "conflicts_detected": ["q1"], "overrides_triggered": [], "noisy_evicted": ["q1"]},
        ]
        with open(os.path.join(tmpdir, "cbet_hotpotqa.json"), "w", encoding="utf-8") as f:
            json.dump(cbet_data, f)

        methods = _find_results(tmpdir, dataset="hotpotqa")
        assert "cbet_hotpotqa" in methods
        m = methods["cbet_hotpotqa"]
        assert m["n"] == 2
        assert m["em"] == 50.0
        assert m["f1"] == 50.0
        assert m["avg_retrieval_rounds"] == 2.5
        assert m["conflict_detected_rate"] == 50.0
        assert m["noisy_branch_evicted_rate"] == 50.0


def test_find_results_dataset_filter():
    """Verify dataset filter excludes non-matching files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cbet_hotpot = [{"qid": "a", "answer": "X", "gold_answer": "X", "em": 1, "f1": 1.0,
                         "iterations": 1, "dag_size": 1, "final_cs": 0.9,
                         "conflicts_detected": [], "overrides_triggered": [], "noisy_evicted": []}]
        with open(os.path.join(tmpdir, "cbet_hotpotqa.json"), "w", encoding="utf-8") as f:
            json.dump(cbet_hotpot, f)
        with open(os.path.join(tmpdir, "cbet_musique.json"), "w", encoding="utf-8") as f:
            json.dump(cbet_hotpot, f)

        methods = _find_results(tmpdir, dataset="musique")
        assert "cbet_musique" in methods
        assert "cbet_hotpotqa" not in methods


def test_find_results_corrupt_file():
    """Corrupt JSON should be skipped with warning, not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "bad.json"), "w", encoding="utf-8") as f:
            f.write("not valid json {{{")
        methods = _find_results(tmpdir, dataset=None)
        assert "bad" not in methods


# ── end-to-end: aggregate + table printing (smoke test) ───────────────────────

def test_aggregate_to_print_table():
    """Verify all required CBET metrics are present in aggregate output."""
    metrics = aggregate(SAMPLE_LOGS)
    required_keys = [
        "n", "em", "f1", "avg_retrieval_rounds", "avg_lm_calls",
        "avg_cs_at_stop", "conflict_detected_rate", "override_triggered_rate",
        "noisy_branch_evicted_rate",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"
    # Verify values are sane
    assert 0 <= metrics["em"] <= 100
    assert 0 <= metrics["f1"] <= 100
    assert metrics["avg_retrieval_rounds"] >= 1
    assert metrics["avg_lm_calls"] >= 1
    assert 0 <= metrics["avg_cs_at_stop"] <= 1
    assert 0 <= metrics["conflict_detected_rate"] <= 100
    assert 0 <= metrics["override_triggered_rate"] <= 100
    assert 0 <= metrics["noisy_branch_evicted_rate"] <= 100
