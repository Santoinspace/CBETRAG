"""Tests for Task 1: data_adapter.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.data_adapter import load_dataset, Question


@pytest.fixture(scope="module")
def hotpot():
    return load_dataset("hotpotqa", n_samples=5)

@pytest.fixture(scope="module")
def musique():
    return load_dataset("musique", n_samples=10)

@pytest.fixture(scope="module")
def wiki():
    return load_dataset("2wikimultihopqa", n_samples=5)


def test_hotpotqa_structure(hotpot):
    q = hotpot[0]
    assert isinstance(q, Question)
    assert q.dataset == "hotpotqa"
    assert q.hop_count == 2
    assert q.query
    assert q.answer
    assert len(q.gold_passages) >= 1
    assert len(q.distractor_passages) >= 1


def test_musique_hop_count(musique):
    import re
    for q in musique:
        m = re.match(r'^(\d+)hop', q.qid)
        if m:
            assert q.hop_count == int(m.group(1)), f"{q.qid}: got {q.hop_count}"


def test_musique_structure(musique):
    q = musique[0]
    assert q.dataset == "musique"
    assert q.answer
    assert q.gold_passages  # musique always has supporting passages


def test_2wiki_structure(wiki):
    q = wiki[0]
    assert q.dataset == "2wikimultihopqa"
    assert q.hop_count == 2
    assert q.query
    assert q.answer


def test_gold_distractor_disjoint(hotpot):
    q = hotpot[0]
    gold_set = set(q.gold_passages)
    dist_set = set(q.distractor_passages)
    assert gold_set.isdisjoint(dist_set)


def test_n_samples(hotpot):
    assert len(hotpot) == 5
