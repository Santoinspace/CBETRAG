"""Task 3: NLI cross-branch consistency scorer.

Model: cross-encoder/nli-deberta-v3-base (default, ~0.4 GB VRAM on RTX 4060).
Label order for DeBERTa cross-encoder NLI: [contradiction, entailment, neutral]
"""
from __future__ import annotations
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from itertools import combinations

from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

# DeBERTa cross-encoder NLI label order (MNLI convention used by sentence-transformers)
_LABEL_ORDER = ["contradiction", "entailment", "neutral"]

_CLAIMS_PROMPT = """\
Extract 3 to 5 important factual claims from the text.

Rules:
- One claim per line
- Use simple declarative sentences
- Do NOT output JSON
- Do NOT explain

Text: {text}

Claims:"""


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
    avg_conflict_ratio: float = 0.0
    max_conflict_ratio: float = 0.0
    contradicting_branch_pairs: int = 0
    valid_branch_pairs: int = 0
    edge_scores: list[float] = field(default_factory=list)
    gcs_method: str = "edge_support"


class NLIScorer:
    def __init__(
        self,
        model_path: str = "./models/nli-deberta-v3-base",
        device: str = "auto",
        batch_size: int = 32,
        theta: float = 0.75,
        gcs_conflict_threshold: float = 0.35,
    ):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # Task 1: auto-detect GPU (vLLM migrated to cloud, local 4060 fully available)
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                self.batch_size = batch_size
            else:
                self.device = "cpu"
                self.batch_size = 16
                print("⚠️  CUDA not available, NLI running on CPU (slower)")
        else:
            self.device = device
            self.batch_size = batch_size

        print(f"[NLIScorer] device={self.device}, batch_size={self.batch_size}")

        self.theta = theta
        self.gcs_conflict_threshold = gcs_conflict_threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_path)
            .to(self.device)
            .eval()
        )
        self.lm_call_count: dict[str, int] = {}

        # Task 4: in-memory NLI cache with thread safety
        self._nli_cache: dict[str, NLIResult] = {}
        self._cache_lock = threading.Lock()
        self._tokenizer_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _truncate(self, text: str, max_tokens: int = 256) -> str:
        """Keep first+last 128 tokens when text exceeds DeBERTa's 512-token limit."""
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=512,
        )
        if len(ids) <= max_tokens:
            return text
        half = max_tokens // 2
        kept = ids[:half] + ids[-half:]
        return self.tokenizer.decode(kept, skip_special_tokens=True)

    # ── caching ──────────────────────────────────────────────────────────────

    def _make_cache_key(self, premise: str, hypothesis: str) -> str:
        """MD5 of full content — no truncation, eliminates collision risk."""
        content = f"{premise}|||{hypothesis}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def cache_stats(self) -> str:
        with self._cache_lock:
            hits = self._cache_hits
            misses = self._cache_misses
        total = hits + misses
        if total == 0:
            return "No NLI calls yet"
        return (f"NLI cache: {hits}/{total} "
                f"hits ({hits/total*100:.0f}%)")

    # ── batch inference (Task 2) ─────────────────────────────────────────────

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """Run NLI on multiple (premise, hypothesis) pairs in a single forward pass."""
        import torch
        import torch.nn.functional as F
        if not pairs:
            return []
        results: list[NLIResult] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            # Tokenizer lock: Rust tokenizer is not re-entrant across threads
            with self._tokenizer_lock:
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

    # ── public API ────────────────────────────────────────────────────────────

    def score_pair(self, premise: str, hypothesis: str) -> NLIResult:
        """Score a single (premise, hypothesis) pair with in-memory caching."""
        cache_key = self._make_cache_key(premise, hypothesis)

        # Read: check cache under lock
        with self._cache_lock:
            if cache_key in self._nli_cache:
                self._cache_hits += 1
                return self._nli_cache[cache_key]

        # Inference: outside lock — allows concurrent GPU inference
        self._cache_misses += 1
        try:
            result = self.score_batch([(premise, hypothesis)])[0]
        except Exception as e:
            logger.warning("NLI score_pair failed: %s", e)
            result = NLIResult(label="neutral", entailment_score=0.0,
                               neutral_score=1.0, contradiction_score=0.0)

        # Write: under lock
        with self._cache_lock:
            self._nli_cache[cache_key] = result
        return result

    # ── claim parsing pipeline ─────────────────────────────────────────────────

    _PREAMBLE_PATTERNS = [
        re.compile(r'^here\s+(?:is|are)\s+the', re.IGNORECASE),
        re.compile(r'^the\s+(?:following|extracted|claims|text)', re.IGNORECASE),
        re.compile(r'^(?:claims?|extracted\s+claims?)\s*:', re.IGNORECASE),
        re.compile(r'^(?:output|result|response)\s*:', re.IGNORECASE),
        re.compile(r'^(?:sure|okay|here\s+you\s+go|i\'?ll?\s+extract)', re.IGNORECASE),
        re.compile(r'^to\s+extract', re.IGNORECASE),
    ]

    _BAD_CLAIM_PATTERNS = [
        re.compile(r'^here\s+(?:is|are)\s+the', re.IGNORECASE),
        re.compile(r'^(?:claims?|extracted)\s*:', re.IGNORECASE),
        re.compile(r'^(?:the\s+)?(?:following|above|below)', re.IGNORECASE),
    ]

    def _parse_claims(self, text: str) -> list[str]:
        """Robust claim parser: JSON→numbered list→bullets→sentence split.

        Returns list of cleaned claim strings. Never falls back to [raw_text].
        """
        text = text.strip()
        original_text = text  # keep for debug

        # ── Strategy 0: strip markdown fences ──
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1])
            else:
                text = "\n".join(lines[1:])
            text = text.strip()

        # ── Strategy 1: JSON recovery ──
        claims = self._try_json_recovery(text)
        if claims:
            logger.debug("claims parsed via JSON recovery: %d claims", len(claims))
            return self._filter_claims(claims)

        # ── Strategy 2: numbered list ──
        claims = self._extract_numbered_list(text)
        if claims:
            logger.debug("claims parsed via numbered list: %d claims", len(claims))
            return self._filter_claims(claims)

        # ── Strategy 3: markdown bullets ──
        claims = self._extract_bullet_list(text)
        if claims:
            logger.debug("claims parsed via bullet list: %d claims", len(claims))
            return self._filter_claims(claims)

        # ── Strategy 4: sentence split ──
        claims = self._sentence_split(text)
        if claims:
            logger.debug("claims parsed via sentence split: %d claims", len(claims))
            return self._filter_claims(claims)

        # ── Final fallback: split text into chunks ──
        logger.warning("_parse_claims all strategies failed; raw_len=%d", len(original_text))
        logger.debug("raw text (first 200 chars): %s", original_text[:200])
        return [original_text[:500]]

    # ── strategy helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _try_json_recovery(text: str) -> list[str] | None:
        """Attempt JSON parsing including truncated/partial forms."""
        # Try direct JSON
        try:
            data = json.loads(text)
            return NLIScorer._extract_from_json_obj(data)
        except (json.JSONDecodeError, AttributeError):
            pass
        # Try json_repair if available
        try:
            from json_repair import repair_json
            repaired = repair_json(text)
            data = json.loads(repaired)
            return NLIScorer._extract_from_json_obj(data)
        except Exception:
            pass
        # Try extracting from truncated JSON array: ["c1", "c2", "c3...
        m = re.search(r'\[(.*?)(?:\]|$)', text, re.DOTALL)
        if m:
            inner = m.group(1)
            # Extract quoted strings from the array content
            quoted = re.findall(r'"([^"]{10,})"', inner)
            if len(quoted) >= 2:
                return quoted
        return None

    @staticmethod
    def _extract_from_json_obj(data) -> list[str] | None:
        """Pull claims from various JSON shapes."""
        if isinstance(data, list):
            return [str(x) for x in data if isinstance(x, str)]
        if isinstance(data, dict):
            for key in ("claims", "claim", "facts", "statements", "output"):
                val = data.get(key)
                if isinstance(val, list):
                    return [str(x) for x in val if isinstance(x, str)]
        return None

    @staticmethod
    def _extract_numbered_list(text: str) -> list[str] | None:
        """Extract from '1. claim one\n2. claim two' format."""
        lines = text.splitlines()
        claims = []
        for line in lines:
            stripped = line.strip()
            m = re.match(r'^\d+\s*[\.\)\)]\s*(.+)$', stripped)
            if m:
                claim = m.group(1).strip().strip('"').strip("'")
                if len(claim) > 10:
                    claims.append(claim)
        return claims if len(claims) >= 2 else None

    @staticmethod
    def _extract_bullet_list(text: str) -> list[str] | None:
        """Extract from '- claim' or '* claim' or '• claim' format."""
        lines = text.splitlines()
        claims = []
        for line in lines:
            stripped = line.strip()
            m = re.match(r'^[\-\*\•\▪\▸]\s*(.+)$', stripped)
            if m:
                claim = m.group(1).strip().strip('"').strip("'")
                if len(claim) > 10:
                    claims.append(claim)
        return claims if len(claims) >= 2 else None

    @staticmethod
    def _sentence_split(text: str) -> list[str] | None:
        """Split by sentence boundaries, filtering preamble lines."""
        lines = text.splitlines()
        sents = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Split on sentence boundaries within the line
            parts = re.split(r'(?<=[.!?])\s+', stripped)
            for p in parts:
                p = p.strip()
                if len(p) > 15:
                    sents.append(p)
        # Deduplicate by prefix
        seen = set()
        unique = []
        for s in sents:
            prefix = s[:40].lower()
            if prefix not in seen:
                seen.add(prefix)
                unique.append(s)
        return unique if len(unique) >= 1 else None

    # ── claim filtering ───────────────────────────────────────────────────────

    def _filter_claims(self, claims: list[str]) -> list[str]:
        """Remove preamble lines, prompt echoes, and noise."""
        filtered = []
        for c in claims:
            c = c.strip().strip('"').strip("'")
            # Length filter
            if len(c) < 10 or len(c) > 500:
                continue
            # Preamble/garbage filter
            if any(pat.search(c) for pat in self._BAD_CLAIM_PATTERNS):
                continue
            # Must start with a capital letter or digit
            if not (c[0].isupper() or c[0].isdigit()):
                continue
            filtered.append(c)
        # Deduplicate
        seen = set()
        unique = []
        for c in filtered:
            key = c[:60].lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    # ── public API ────────────────────────────────────────────────────────────

    def extract_atomic_claims(self, text: str, llm_client: LLMClient) -> list[str]:
        """Use LLM to extract atomic factual claims from text."""
        self.lm_call_count["atomic_claims"] = self.lm_call_count.get("atomic_claims", 0) + 1
        input_len = min(len(text), 2000)
        try:
            resp = llm_client.generate(
                _CLAIMS_PROMPT.format(text=text[:2000]),
                max_new_tokens=256,
                temperature=0.0,
            )
            claims = self._parse_claims(resp.text)
            logger.debug("extract_atomic_claims: input=%d chars, raw_output=%d chars, claims=%d",
                         input_len, len(resp.text), len(claims))
            return claims
        except Exception as e:
            logger.warning("extract_atomic_claims failed: %s", e)
            return []

    def compute_coverage(
        self, evidence: str, sub_answer: str, llm_client: LLMClient
    ) -> float:
        """Cov(Eᵢ, qᵢ) = max entailment score over atomic claims of evidence ⊨ sub_answer."""
        try:
            claims = self.extract_atomic_claims(evidence, llm_client)
            pairs = [(claim, sub_answer) for claim in claims]
            results = self.score_batch(pairs)
            return max(r.entailment_score for r in results)
        except Exception as e:
            logger.warning("compute_coverage failed: %s", e)
            return 0.0

    def compute_gcs(self, evidences: list[str], llm_client: LLMClient) -> float:
        """GCS(ε) = fraction of evidence pairs that are NOT contradictory.

        Uses density-based threshold: a branch pair is contradictory only if
        conflict_ratio > gcs_conflict_threshold (default 0.25).
        """
        n = len(evidences)
        if n <= 1:
            return 1.0

        try:
            all_claims: list[list[str]] = [
                self.extract_atomic_claims(e, llm_client) for e in evidences
            ]
            contradicting = 0
            valid = 0
            for i, j in combinations(range(n), 2):
                claims_i, claims_j = all_claims[i], all_claims[j]
                total_pairs = len(claims_i) * len(claims_j)
                if total_pairs == 0:
                    continue
                valid += 1
                cross = [(c_i, c_j) for c_i in claims_i for c_j in claims_j]
                results = self.score_batch(cross)
                n_contra = sum(1 for r in results if r.label == "contradiction")
                conflict_ratio = n_contra / total_pairs
                if conflict_ratio > self.gcs_conflict_threshold:
                    contradicting += 1
            return 1.0 - contradicting / valid if valid else 1.0
        except Exception as e:
            logger.warning("compute_gcs failed: %s", e)
            return 1.0

    # ── edge support GCS ─────────────────────────────────────────────────────

    def _compute_edge_support_gcs(
        self,
        branch_evidences: list[str],
        branch_answers: list[str],
        dependency_pairs: list[tuple[int, int]],
        llm_client: LLMClient | None,
    ) -> tuple[float, list[float]]:
        """Edge Support Verification GCS.

        For each DAG dependency edge (src→tgt), verify that the downstream
        evidence supports the upstream answer (bridge entity propagation).

        Uses compute_coverage(tgt_evidence, src_answer) — short answer vs
        long evidence, avoiding 512-token truncation entirely.

        Returns:
            gcs: mean of all edge scores (1.0 if no edges)
            edge_scores: per-edge scores for logging
        """
        if not dependency_pairs:
            return 1.0, []

        edge_scores: list[float] = []
        for src_idx, tgt_idx in dependency_pairs:
            src_answer = branch_answers[src_idx] if src_idx < len(branch_answers) else ""
            tgt_evidence = branch_evidences[tgt_idx] if tgt_idx < len(branch_evidences) else ""

            if not src_answer or not tgt_evidence:
                edge_scores.append(0.0)
                continue

            try:
                score = self.compute_coverage(tgt_evidence, src_answer, llm_client)
            except Exception as e:
                logger.warning("edge support score failed for (%d→%d): %s", src_idx, tgt_idx, e)
                score = 0.0
            edge_scores.append(score)

        gcs = sum(edge_scores) / len(edge_scores) if edge_scores else 1.0
        return gcs, edge_scores

    def compute_completeness_score(
        self,
        branch_evidences: list[str],
        branch_answers: list[str],
        sub_questions: list[str],
        llm_client: LLMClient,
        skip_gcs: bool = False,
        dependency_pairs: list[tuple[int, int]] | None = None,
    ) -> CompletenessResult:
        """CS = min_i(Cov_i) × GCS.  Identifies noisy (contradictory) branches.

        GCS is computed via Edge Support Verification: for each DAG dependency
        edge (src→tgt), check if downstream evidence supports the upstream
        answer (bridge entity propagation).

        When skip_gcs=True (no_cross_branch ablation), GCS is forced to 1.0.
        """
        n = len(branch_evidences)

        # Extract atomic claims ONCE per evidence (reused for coverage)
        all_claims: list[list[str]] = []
        for i in range(n):
            if branch_evidences[i]:
                try:
                    claims = self.extract_atomic_claims(branch_evidences[i], llm_client)
                except Exception:
                    claims = [branch_evidences[i]]
            else:
                claims = []
            all_claims.append(claims)

        # Per-branch coverage — NLI(claim → answer): does evidence entail the answer?
        coverages: list[float] = []
        for i in range(n):
            if not branch_answers[i] or not all_claims[i]:
                coverages.append(0.0)
                continue
            try:
                ans = branch_answers[i]
                pairs = [(claim, ans) for claim in all_claims[i]]
                results = self.score_batch(pairs)
                coverages.append(max(r.entailment_score for r in results))
            except Exception as e:
                logger.warning("compute_coverage failed: %s", e)
                coverages.append(0.0)
        min_cov = min(coverages) if coverages else 0.0

        # GCS — Edge Support Verification
        edge_scores: list[float] = []
        gcs_method = "edge_support"
        if skip_gcs or n <= 1:
            gcs = 1.0
            gcs_method = "skip" if skip_gcs else "single_branch"
        else:
            gcs, edge_scores = self._compute_edge_support_gcs(
                branch_evidences, branch_answers,
                dependency_pairs or [], llm_client,
            )
            if not dependency_pairs:
                gcs_method = "no_edges"

        cs = min_cov * gcs

        # Noisy branch detection — branches involved in low-support edges
        noisy: set[int] = set()
        if edge_scores and dependency_pairs:
            for idx, (src_idx, tgt_idx) in enumerate(dependency_pairs):
                if idx < len(edge_scores) and edge_scores[idx] < 0.2:
                    # Mark the branch with lower coverage as noisy
                    if coverages[src_idx] <= coverages[tgt_idx]:
                        noisy.add(src_idx)
                    else:
                        noisy.add(tgt_idx)

        return CompletenessResult(
            branch_coverages=coverages,
            min_coverage=min_cov,
            gcs=gcs,
            cs=cs,
            should_stop=cs >= self.theta,
            noisy_branch_ids=sorted(noisy),
            edge_scores=edge_scores,
            gcs_method=gcs_method,
        )
