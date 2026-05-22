"""Tests for Task 2: dag_extractor.py — uses mock LLMClient, no GPU required."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
from src.llm_client import LLMClient, LLMResponse
from src.dag_extractor import extract_dag, QuestionDAG, SubQuestion


class MockLLM(LLMClient):
    def __init__(self, response: str):
        self._response = response

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(text=self._response)


def _make_json(sqs: list[dict]) -> str:
    return json.dumps({"sub_questions": sqs})


# ── fixtures ──────────────────────────────────────────────────────────────────

HOTPOT_QUERY = (
    "What nationality was the director of the film that featured "
    "the song 'My Heart Will Go On'?"
)

MUSIQUE_4HOP_QUERY = (
    "Who was the spouse of the person who founded the company "
    "that acquired the studio that produced the film directed by "
    "the person born in Whitby?"
)

HOTPOT_SQS = [
    {"id": "q1", "text": "Which film featured the song 'My Heart Will Go On'?", "depends_on": []},
    {"id": "q2", "text": "Who directed [answer of q1]?", "depends_on": ["q1"]},
    {"id": "q3", "text": "What is the nationality of [answer of q2]?", "depends_on": ["q2"]},
]

MUSIQUE_SQS = [
    {"id": "q1", "text": "Who was born in Whitby?", "depends_on": []},
    {"id": "q2", "text": "What film did [answer of q1] direct?", "depends_on": ["q1"]},
    {"id": "q3", "text": "Which studio produced [answer of q2]?", "depends_on": ["q2"]},
    {"id": "q4", "text": "Which company acquired [answer of q3]?", "depends_on": ["q3"]},
    {"id": "q5", "text": "Who founded [answer of q4]?", "depends_on": ["q4"]},
    {"id": "q6", "text": "Who was the spouse of [answer of q5]?", "depends_on": ["q5"]},
]

PARALLEL_SQS = [
    {"id": "q1", "text": "Where was person A born?", "depends_on": []},
    {"id": "q2", "text": "Where was person B born?", "depends_on": []},
    {"id": "q3", "text": "Which city is between [answer of q1] and [answer of q2]?", "depends_on": ["q1", "q2"]},
]


# ── tests ─────────────────────────────────────────────────────────────────────

def test_hotpotqa_min_subquestions():
    dag = extract_dag(HOTPOT_QUERY, MockLLM(_make_json(HOTPOT_SQS)))
    assert len(dag.sub_questions) >= 2


def test_musique_parallel_leaves():
    """4-hop+ MuSiQue: parallel leaf nodes exist."""
    dag = extract_dag(MUSIQUE_4HOP_QUERY, MockLLM(_make_json(PARALLEL_SQS)))
    assert len(dag.get_leaves()) >= 2


def test_dag_no_cycle():
    """Topological sort must not produce a cycle."""
    dag = extract_dag(HOTPOT_QUERY, MockLLM(_make_json(HOTPOT_SQS)))
    order = dag.get_execution_order()
    seen: set[str] = set()
    for batch in order:
        for sq in batch:
            # All dependencies must already be resolved
            assert all(d in seen for d in sq.depends_on), (
                f"{sq.id} depends on {sq.depends_on} but seen={seen}"
            )
        seen.update(sq.id for sq in batch)


def test_leaves_have_no_depends():
    dag = extract_dag(HOTPOT_QUERY, MockLLM(_make_json(HOTPOT_SQS)))
    for leaf in dag.get_leaves():
        assert leaf.depends_on == []


def test_execution_order_respects_deps():
    dag = extract_dag("test", MockLLM(_make_json(PARALLEL_SQS)))
    order = dag.get_execution_order()
    # q1 and q2 should be in the first batch (parallel)
    first_ids = {sq.id for sq in order[0]}
    assert "q1" in first_ids and "q2" in first_ids
    # q3 must come after
    last_ids = {sq.id for sq in order[-1]}
    assert "q3" in last_ids


def test_fallback_on_bad_json():
    """Invalid JSON → fallback single-node DAG."""
    dag = extract_dag("some query", MockLLM("not json at all !!!"))
    assert len(dag.sub_questions) == 1
    assert dag.sub_questions[0].text == "some query"
    assert dag.sub_questions[0].is_leaf is True


def test_markdown_fenced_json():
    """LLM wraps JSON in ```json ... ``` — should still parse."""
    raw = "```json\n" + _make_json(HOTPOT_SQS) + "\n```"
    dag = extract_dag(HOTPOT_QUERY, MockLLM(raw))
    assert len(dag.sub_questions) == len(HOTPOT_SQS)
