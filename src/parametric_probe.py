"""Task 4: Parametric memory probe — answers sub-questions without retrieval context."""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field

from src.llm_client import LLMClient
from src.nli_scorer import NLIScorer

logger = logging.getLogger(__name__)

_PROBE_PROMPT = "Answer this question concisely: {sub_question}"

# Confidence threshold: entropy < 0.3 → model is certain → real conflict
_CERTAINTY_THRESHOLD = 0.3


@dataclass
class ParametricMemory:
    answer: str
    confidence: float       # Mean Token Entropy (lower = more certain)
    raw_logprobs: list[float] = field(default_factory=list)


@dataclass
class ConflictResult:
    has_conflict: bool
    conflict_type: str      # "parametric_vs_retrieved" / "no_conflict" / "uncertain"
    parametric_answer: str
    retrieved_answer: str
    trust_retrieved: float  # GCS × 𝟙[conflict]


def _mean_token_entropy(logprobs: list[float]) -> float:
    """Mean Token Entropy from per-token log-probabilities.

    H = -mean_i( sum_w p(w|x<i) log p(w|x<i) )
    For a greedy/argmax token, the single-token entropy lower-bound is:
        H_i ≈ -logprob_i  (since p_chosen ≈ exp(logprob_i))
    This is the standard approximation used when only top-1 logprobs are available.
    """
    if not logprobs:
        return 1.0  # maximum uncertainty when no logprobs
    # -logprob_i is the per-token surprisal; mean surprisal ≈ mean entropy
    return float(-sum(logprobs) / len(logprobs))


def _extract_answer_from_evidence(evidence: str, llm_client: LLMClient | None = None) -> str:
    """Extract the key factual answer from retrieved evidence (no LLM call needed).

    Uses the first substantive sentence of the evidence as the extracted answer.
    The actual conflict detection relies on NLI(evidence, parametric_answer), not
    on this extracted string — this is primarily for logging.
    """
    if not evidence:
        return ""
    # Return first sentence up to 200 chars
    first_sent = evidence.split(".")[0].strip()
    return first_sent[:200] if first_sent else evidence[:200]


class ParametricProbe:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.lm_call_count: dict[str, int] = {}

    def probe(self, sub_question: str) -> ParametricMemory:
        """Answer sub_question using only parametric knowledge (no retrieval context)."""
        self.lm_call_count["parametric_probe"] = self.lm_call_count.get("parametric_probe", 0) + 1
        try:
            resp = self.llm.generate(
                _PROBE_PROMPT.format(sub_question=sub_question),
                max_new_tokens=128,
                temperature=0.0,
            )
            confidence = _mean_token_entropy(resp.logprobs)
            return ParametricMemory(
                answer=resp.text.strip(),
                confidence=confidence,
                raw_logprobs=resp.logprobs,
            )
        except Exception as e:
            logger.warning("parametric probe failed: %s", e)
            return ParametricMemory(answer="", confidence=1.0, raw_logprobs=[])

    def detect_conflict(
        self,
        parametric: ParametricMemory,
        retrieved_evidence: str,
        nli_scorer: NLIScorer,
        llm_client: LLMClient,
        gcs: float = 1.0,
    ) -> ConflictResult:
        """Detect conflict between parametric memory and retrieved evidence.

        A conflict is "real" only when the model is certain (confidence < threshold).
        trust_retrieved = GCS × 𝟙[conflict]
        """
        retrieved_answer = _extract_answer_from_evidence(retrieved_evidence)

        if not parametric.answer:
            return ConflictResult(
                has_conflict=False,
                conflict_type="uncertain",
                parametric_answer="",
                retrieved_answer=retrieved_answer,
                trust_retrieved=0.0,
            )

        try:
            nli_result = nli_scorer.score_pair(
                premise=retrieved_evidence,
                hypothesis=parametric.answer,
            )
            is_contradiction = nli_result.label == "contradiction"
        except Exception as e:
            logger.warning("NLI in detect_conflict failed: %s", e)
            is_contradiction = False

        # Only treat as real conflict when model is certain about its parametric answer
        model_certain = parametric.confidence < _CERTAINTY_THRESHOLD

        if is_contradiction and model_certain:
            conflict_type = "parametric_vs_retrieved"
            has_conflict = True
        elif is_contradiction:
            conflict_type = "uncertain"   # contradiction but model wasn't sure anyway
            has_conflict = False
        else:
            conflict_type = "no_conflict"
            has_conflict = False

        trust_retrieved = gcs if has_conflict else 0.0

        return ConflictResult(
            has_conflict=has_conflict,
            conflict_type=conflict_type,
            parametric_answer=parametric.answer,
            retrieved_answer=retrieved_answer,
            trust_retrieved=trust_retrieved,
        )
