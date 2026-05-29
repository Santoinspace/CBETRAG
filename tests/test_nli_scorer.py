"""Tests for Task 3: nli_scorer.py.

Fast tests use a mock NLI model (no GPU needed).
Integration tests (marked 'gpu') require the real DeBERTa model.
Run fast tests only: pytest tests/test_nli_scorer.py -m "not gpu"
"""
import sys, json, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from unittest.mock import MagicMock, patch
from src.nli_scorer import NLIScorer, NLIResult, CompletenessResult
from src.llm_client import LLMClient, LLMResponse


# ── helpers ───────────────────────────────────────────────────────────────────

class MockLLM(LLMClient):
    """Returns a fixed claims JSON."""
    def __init__(self, claims: list[str]):
        import tempfile
        super().__init__(cache_dir=tempfile.mkdtemp(prefix="test_llm_cache_"))
        self._resp = json.dumps({"claims": claims})
    def _generate(self, prompt, max_new_tokens=512, temperature=0.0, **kw):
        return LLMResponse(text=self._resp)


def _make_scorer(label_probs: dict[str, list[float]]) -> NLIScorer:
    """Build NLIScorer with mocked model that returns fixed probabilities per text pair."""
    scorer = NLIScorer.__new__(NLIScorer)
    scorer.device = "cpu"
    scorer.batch_size = 16
    scorer.theta = 0.75
    scorer.gcs_conflict_threshold = 0.35
    scorer.lm_call_count = {}
    # Cache attributes
    scorer._nli_cache = {}
    scorer._cache_lock = threading.Lock()
    scorer._tokenizer_lock = threading.Lock()
    scorer._cache_hits = 0
    scorer._cache_misses = 0

    # Mock tokenizer
    tok = MagicMock()
    tok.encode.return_value = list(range(10))
    tok.decode.return_value = "text"
    enc_out = MagicMock()
    enc_out.to.return_value = enc_out
    tok.return_value = enc_out
    scorer.tokenizer = tok

    # Mock model: always returns the same logits
    # label order: [contradiction, entailment, neutral]
    model = MagicMock()
    model.eval.return_value = model

    def fake_forward(**kwargs):
        # Return logits shaped (B, 3) based on default_probs
        B = 1
        out = MagicMock()
        out.logits = torch.tensor([label_probs["default"]])
        return out

    model.return_value = fake_forward()
    model.__call__ = lambda self_, **kw: fake_forward(**kw)
    scorer.model = model

    return scorer


def _scorer_with_logits(logits_fn) -> NLIScorer:
    """NLIScorer whose score_batch is fully replaced."""
    scorer = NLIScorer.__new__(NLIScorer)
    scorer.device = "cpu"
    scorer.batch_size = 16
    scorer.theta = 0.75
    scorer.gcs_conflict_threshold = 0.35
    scorer.lm_call_count = {}
    # Cache attributes
    scorer._nli_cache = {}
    scorer._cache_lock = threading.Lock()
    scorer._tokenizer_lock = threading.Lock()
    scorer._cache_hits = 0
    scorer._cache_misses = 0
    tok = MagicMock()
    tok.encode.return_value = list(range(10))
    tok.decode.return_value = "text"
    scorer.tokenizer = tok
    scorer.score_batch = logits_fn
    return scorer


# ── unit tests (no GPU) ───────────────────────────────────────────────────────

def test_nliresult_label_fields():
    r = NLIResult(label="entailment", entailment_score=0.9,
                  neutral_score=0.05, contradiction_score=0.05)
    assert r.label == "entailment"
    assert r.entailment_score == pytest.approx(0.9)


def test_parse_claims_valid_json():
    scorer = _scorer_with_logits(lambda pairs: [])
    claims = scorer._parse_claims(
        '{"claims": ["Alpha is the first letter", "Beta is the second letter"]}')
    assert claims == ["Alpha is the first letter", "Beta is the second letter"]


def test_parse_claims_markdown_fence():
    scorer = _scorer_with_logits(lambda pairs: [])
    raw = '```json\n{"claims": ["X-ray imaging is a medical technique"]}\n```'
    assert scorer._parse_claims(raw) == ["X-ray imaging is a medical technique"]


def test_parse_claims_fallback():
    scorer = _scorer_with_logits(lambda pairs: [])
    # Purely unstructured text — falls through to sentence split or final fallback
    claims = scorer._parse_claims(
        "This is a valid claim sentence that meets the length filter. "
        "The Berlin Wall fell in November 1989.")
    assert len(claims) >= 1  # At least one claim from sentence split


def test_gcs_single_branch():
    """n=1 → GCS must be 1.0 (no pairs to compare)."""
    scorer = _scorer_with_logits(lambda pairs: [])
    llm = MockLLM(["some claim"])
    assert scorer.compute_gcs(["evidence A"], llm) == 1.0


def test_gcs_consistent_evidences():
    """Two consistent evidences → no contradiction → GCS = 1.0."""
    def always_entailment(pairs):
        return [NLIResult("entailment", 0.9, 0.05, 0.05) for _ in pairs]

    scorer = _scorer_with_logits(always_entailment)
    llm = MockLLM(["claim 1", "claim 2"])
    gcs = scorer.compute_gcs(["evidence A", "evidence B"], llm)
    assert gcs == pytest.approx(1.0)


def test_gcs_contradictory_evidences():
    """All pairs contradict → GCS = 0.0 (conflict_ratio=1.0 > 0.35)."""
    def always_contradiction(pairs):
        return [NLIResult("contradiction", 0.05, 0.05, 0.9) for _ in pairs]

    scorer = _scorer_with_logits(always_contradiction)
    llm = MockLLM(["Claim one is long enough claim", "Claim two is also long enough claim"])
    gcs = scorer.compute_gcs(["evidence A", "evidence B"], llm)
    assert gcs == pytest.approx(0.0)


def test_coverage_high_for_relevant_evidence():
    """Evidence that entails the answer → Cov > 0.7."""
    def high_entailment(pairs):
        return [NLIResult("entailment", 0.92, 0.04, 0.04) for _ in pairs]

    scorer = _scorer_with_logits(high_entailment)
    llm = MockLLM(["The Berlin Wall fell in 1989."])
    cov = scorer.compute_coverage(
        "The Berlin Wall fell on November 9, 1989.",
        "The Berlin Wall fell in 1989.",
        llm,
    )
    assert cov > 0.7


def test_coverage_low_for_irrelevant_evidence():
    """Unrelated evidence → Cov < 0.3."""
    def low_entailment(pairs):
        return [NLIResult("neutral", 0.05, 0.9, 0.05) for _ in pairs]

    scorer = _scorer_with_logits(low_entailment)
    llm = MockLLM(["Paris is the capital of France."])
    cov = scorer.compute_coverage(
        "Paris is the capital of France.",
        "The Berlin Wall fell in 1989.",
        llm,
    )
    assert cov < 0.3


def test_completeness_score_structure():
    def mixed(pairs):
        return [NLIResult("entailment", 0.8, 0.1, 0.1) for _ in pairs]

    scorer = _scorer_with_logits(mixed)
    llm = MockLLM(["claim"])
    result = scorer.compute_completeness_score(
        branch_evidences=["ev1", "ev2"],
        branch_answers=["ans1", "ans2"],
        sub_questions=["q1", "q2"],
        llm_client=llm,
    )
    assert isinstance(result, CompletenessResult)
    assert len(result.branch_coverages) == 2
    assert 0.0 <= result.cs <= 1.0
    assert result.cs == pytest.approx(result.min_coverage * result.gcs)


def test_completeness_should_stop_when_cs_high():
    def high(pairs):
        return [NLIResult("entailment", 0.95, 0.03, 0.02) for _ in pairs]

    scorer = _scorer_with_logits(high)
    scorer.theta = 0.75
    llm = MockLLM(["Claim one with enough words to pass the length filter"])
    result = scorer.compute_completeness_score(["ev1"], ["ans1"], ["q1"], llm)
    assert result.should_stop is True


def test_completeness_should_not_stop_when_cs_low():
    def low(pairs):
        return [NLIResult("neutral", 0.1, 0.8, 0.1) for _ in pairs]

    scorer = _scorer_with_logits(low)
    scorer.theta = 0.75
    llm = MockLLM(["A claim with enough words to pass the length filter"])
    result = scorer.compute_completeness_score(["ev1"], ["ans1"], ["q1"], llm)
    assert result.should_stop is False


def test_noisy_branch_marked_lower_coverage():
    """When edge support score < 0.2, lower-coverage branch marked noisy."""
    call_count = [0]

    def mixed_scores(pairs):
        call_count[0] += 1
        # Odd calls: coverage (branch 0 gets low entailment, branch 1 gets high)
        # Even calls: edge support (low score → noisy)
        if call_count[0] == 1:
            # Coverage for branch 0: low entailment
            return [NLIResult("entailment", 0.2, 0.1, 0.7) for _ in pairs]
        elif call_count[0] == 2:
            # Coverage for branch 1: high entailment
            return [NLIResult("entailment", 0.9, 0.05, 0.05) for _ in pairs]
        else:
            # Edge support: low entailment → edge_score < 0.2 → noisy
            return [NLIResult("neutral", 0.05, 0.9, 0.05) for _ in pairs]

    scorer = _scorer_with_logits(mixed_scores)
    scorer.theta = 0.75

    llm = MockLLM(["Claim one with enough length here", "Claim two also long enough"])
    result = scorer.compute_completeness_score(
        branch_evidences=["ev_low", "ev_high"],
        branch_answers=["ans1", "ans2"],
        sub_questions=["q1", "q2"],
        llm_client=llm,
        dependency_pairs=[(0, 1)],  # edge from branch 0 to branch 1
    )
    # Branch 0 has lower coverage (0.2 < 0.9), should be marked noisy
    assert 0 in result.noisy_branch_ids


def test_score_pair_fallback_on_error():
    """score_pair returns neutral NLIResult on model failure."""
    scorer = NLIScorer.__new__(NLIScorer)
    scorer.device = "cpu"
    scorer.batch_size = 16
    scorer.theta = 0.75
    tok = MagicMock()
    tok.encode.return_value = list(range(5))
    tok.decode.return_value = "x"
    scorer.tokenizer = tok
    # Cache attributes
    scorer._nli_cache = {}
    scorer._cache_lock = threading.Lock()
    scorer._tokenizer_lock = threading.Lock()
    scorer._cache_hits = 0
    scorer._cache_misses = 0

    def raise_error(pairs):
        raise RuntimeError("GPU OOM")

    scorer.score_batch = raise_error
    result = scorer.score_pair("premise", "hypothesis")
    assert result.label == "neutral"


# ── integration tests (require real model + GPU) ──────────────────────────────

MODEL_PATH = "./models/nli-deberta-v3-base"
_model_available = Path(MODEL_PATH).exists()


@pytest.mark.gpu
@pytest.mark.skipif(not _model_available, reason="DeBERTa model not downloaded")
class TestNLIScorerIntegration:
    @pytest.fixture(scope="class")
    def scorer(self):
        return NLIScorer(model_path=MODEL_PATH, device="auto")

    @pytest.fixture(scope="class")
    def llm(self):
        # Minimal mock that returns the input text as a single claim
        class PassthroughLLM(LLMClient):
            def __init__(self):
                import tempfile
                super().__init__(cache_dir=tempfile.mkdtemp(prefix="test_llm_cache_"))
            def _generate(self, prompt, max_new_tokens=512, temperature=0.0, **kw):
                # Extract text after "Text:" line
                for line in prompt.splitlines():
                    if line.startswith("Text:"):
                        text = line[5:].strip()
                        return LLMResponse(
                            text=json.dumps({"claims": [text]})
                        )
                return LLMResponse(text='{"claims": ["unknown"]}')
        return PassthroughLLM()

    def test_consistent_gcs_near_one(self, scorer, llm):
        ev1 = "The Berlin Wall fell in November 1989."
        ev2 = "The Berlin Wall collapsed in 1989."
        gcs = scorer.compute_gcs([ev1, ev2], llm)
        assert gcs >= 0.5, f"Expected GCS >= 0.5 for consistent evidences, got {gcs}"

    def test_contradictory_gcs_below_half(self, scorer, llm):
        ev1 = "The event happened in 1989."
        ev2 = "The event happened in 1991."
        gcs = scorer.compute_gcs([ev1, ev2], llm)
        assert gcs < 0.5, f"Expected GCS < 0.5 for contradictory evidences, got {gcs}"

    def test_coverage_relevant_above_threshold(self, scorer, llm):
        evidence = "The Berlin Wall fell on November 9, 1989."
        answer = "1989"
        cov = scorer.compute_coverage(evidence, answer, llm)
        assert cov > 0.7, f"Expected Cov > 0.7 for relevant evidence, got {cov}"

    def test_coverage_irrelevant_below_threshold(self, scorer, llm):
        evidence = "The Eiffel Tower was built in Paris between 1887 and 1889."
        answer = "Amsterdam is the capital of the Netherlands"
        cov = scorer.compute_coverage(evidence, answer, llm)
        assert cov < 0.3, f"Expected Cov < 0.3 for irrelevant evidence, got {cov}"
