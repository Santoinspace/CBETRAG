"""Task 3: NLI cross-branch consistency scorer.

Model: cross-encoder/nli-deberta-v3-base (default, ~0.4 GB VRAM on RTX 4060).
Label order for DeBERTa cross-encoder NLI: [contradiction, entailment, neutral]
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from itertools import combinations

import torch
import torch.nn.functional as F

from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

# DeBERTa cross-encoder NLI label order (MNLI convention used by sentence-transformers)
_LABEL_ORDER = ["contradiction", "entailment", "neutral"]

_CLAIMS_PROMPT = """\
Extract all atomic factual claims from this text as a JSON list of simple declarative sentences.
Each claim must be independently verifiable. Remove opinions and vague statements.
Text: {text}
Output: {{"claims": ["claim1", "claim2", ...]}}"""


@dataclass
class NLIResult:
    label: str              # "entailment" / "neutral" / "contradiction"
    entailment_score: float
    neutral_score: float
    contradiction_score: float


@dataclass
class CompletenessResult:
    branch_coverages: list[float]
    min_coverage: float
    gcs: float
    cs: float               # = min_coverage × gcs
    should_stop: bool       # cs >= theta
    noisy_branch_ids: list[int]


class NLIScorer:
    def __init__(
        self,
        model_path: str = "./models/nli-deberta-v3-base",
        device: str = "cuda",
        batch_size: int = 16,
        theta: float = 0.75,
    ):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = device
        self.batch_size = batch_size
        self.theta = theta
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_path)
            .to(device)
            .eval()
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _truncate(self, text: str, max_tokens: int = 256) -> str:
        """Keep first+last 128 tokens when text exceeds DeBERTa's 512-token limit."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        half = max_tokens // 2
        kept = ids[:half] + ids[-half:]
        return self.tokenizer.decode(kept, skip_special_tokens=True)

    def _batch_score(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """Run NLI on a list of (premise, hypothesis) pairs in batches."""
        results: list[NLIResult] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            premises = [self._truncate(p) for p, _ in batch]
            hypotheses = [self._truncate(h) for _, h in batch]
            enc = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits          # (B, 3)
            probs = F.softmax(logits, dim=-1).cpu().tolist()
            for p in probs:
                # p order: [contradiction, entailment, neutral]
                results.append(
                    NLIResult(
                        label=_LABEL_ORDER[int(torch.tensor(p).argmax())],
                        contradiction_score=p[0],
                        entailment_score=p[1],
                        neutral_score=p[2],
                    )
                )
        return results

    def _parse_claims(self, text: str) -> list[str]:
        """Parse JSON claims list from LLM output, fallback to [text]."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            data = json.loads(text)
            claims = data.get("claims", [])
            if isinstance(claims, list) and claims:
                return [str(c) for c in claims]
        except (json.JSONDecodeError, AttributeError):
            pass
        return [text]  # fallback: treat whole response as one claim

    # ── public API ────────────────────────────────────────────────────────────

    def score_pair(self, premise: str, hypothesis: str) -> NLIResult:
        """Score a single (premise, hypothesis) pair."""
        try:
            return self._batch_score([(premise, hypothesis)])[0]
        except Exception as e:
            logger.warning("NLI score_pair failed: %s", e)
            return NLIResult(label="neutral", entailment_score=0.0,
                             neutral_score=1.0, contradiction_score=0.0)

    def extract_atomic_claims(self, text: str, llm_client: LLMClient) -> list[str]:
        """Use LLM to extract atomic factual claims from text."""
        try:
            resp = llm_client.generate(
                _CLAIMS_PROMPT.format(text=text[:2000]),  # cap input length
                max_new_tokens=256,
                temperature=0.0,
            )
            return self._parse_claims(resp.text)
        except Exception as e:
            logger.warning("extract_atomic_claims failed: %s", e)
            return [text]

    def compute_coverage(
        self, evidence: str, sub_answer: str, llm_client: LLMClient
    ) -> float:
        """Cov(Eᵢ, qᵢ) = max entailment score over atomic claims of evidence ⊨ sub_answer."""
        try:
            claims = self.extract_atomic_claims(evidence, llm_client)
            pairs = [(claim, sub_answer) for claim in claims]
            results = self._batch_score(pairs)
            return max(r.entailment_score for r in results)
        except Exception as e:
            logger.warning("compute_coverage failed: %s", e)
            return 0.0

    def compute_gcs(self, evidences: list[str], llm_client: LLMClient) -> float:
        """GCS(ε) = fraction of evidence pairs that are NOT contradictory.

        Uses atomic claims: for each pair (Eᵢ, Eⱼ), extract claims from both,
        then check all cross-claim pairs for contradiction.
        """
        n = len(evidences)
        if n <= 1:
            return 1.0

        try:
            # Extract claims for each branch
            all_claims: list[list[str]] = [
                self.extract_atomic_claims(e, llm_client) for e in evidences
            ]

            contradiction_pairs = 0
            total_pairs = 0

            for i, j in combinations(range(n), 2):
                claims_i, claims_j = all_claims[i], all_claims[j]
                cross_pairs = [(c_i, c_j) for c_i in claims_i for c_j in claims_j]
                if not cross_pairs:
                    total_pairs += 1
                    continue
                results = self._batch_score(cross_pairs)
                has_contradiction = any(r.label == "contradiction" for r in results)
                total_pairs += 1
                if has_contradiction:
                    contradiction_pairs += 1

            return 1.0 - contradiction_pairs / total_pairs if total_pairs else 1.0
        except Exception as e:
            logger.warning("compute_gcs failed: %s", e)
            return 1.0

    def compute_completeness_score(
        self,
        branch_evidences: list[str],
        branch_answers: list[str],
        sub_questions: list[str],
        llm_client: LLMClient,
    ) -> CompletenessResult:
        """CS = min_i(Cov_i) × GCS.  Identifies noisy (contradictory) branches."""
        n = len(branch_evidences)

        # Per-branch coverage
        coverages = [
            self.compute_coverage(branch_evidences[i], branch_answers[i], llm_client)
            if branch_answers[i]
            else 0.0
            for i in range(n)
        ]

        gcs = self.compute_gcs(branch_evidences, llm_client)
        min_cov = min(coverages) if coverages else 0.0
        cs = min_cov * gcs

        # Identify noisy branches: involved in any contradiction → keep lower-coverage one
        noisy: set[int] = set()
        if n > 1:
            try:
                all_claims = [
                    self.extract_atomic_claims(e, llm_client) for e in branch_evidences
                ]
                for i, j in combinations(range(n), 2):
                    cross = [(ci, cj) for ci in all_claims[i] for cj in all_claims[j]]
                    if not cross:
                        continue
                    results = self._batch_score(cross)
                    if any(r.label == "contradiction" for r in results):
                        # Mark the branch with lower coverage as noisy
                        noisy.add(i if coverages[i] <= coverages[j] else j)
            except Exception as e:
                logger.warning("noisy branch detection failed: %s", e)

        return CompletenessResult(
            branch_coverages=coverages,
            min_coverage=min_cov,
            gcs=gcs,
            cs=cs,
            should_stop=cs >= self.theta,
            noisy_branch_ids=sorted(noisy),
        )
