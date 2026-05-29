"""Task 2: DAG sub-question extractor."""
from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

_FAILURE_LOG = Path("experiments/results/dag_failures.log")

_PROMPT = """\
You are a multi-hop question decomposer. Given a complex multi-hop question, \
decompose it into atomic sub-questions with their dependency relationships.

Rules:
1. Each sub-question must be answerable with a single factual retrieval
2. Mark dependencies explicitly (which sub-questions must be answered first)
3. Independent sub-questions have empty depends_on lists
4. Output ONLY valid JSON, no explanation

Question: {query}

Output format:
{{
  "sub_questions": [
    {{"id": "q1", "text": "...", "depends_on": []}},
    {{"id": "q2", "text": "...", "depends_on": []}},
    {{"id": "q3", "text": "... [answer of q1] ... [answer of q2] ...", "depends_on": ["q1", "q2"]}}
  ]
}}"""

_PROMPT_STRICT = """\
You are a multi-hop question decomposer. Output ONLY a JSON object — no markdown, \
no explanation, no extra text.

Question: {query}

Required JSON schema (follow exactly):
{{
  "sub_questions": [
    {{"id": "q1", "text": "<atomic question>", "depends_on": []}},
    {{"id": "q2", "text": "<atomic question>", "depends_on": ["q1"]}}
  ]
}}"""


@dataclass
class SubQuestion:
    id: str
    text: str
    depends_on: list[str] = field(default_factory=list)
    is_leaf: bool = False


@dataclass
class QuestionDAG:
    root_query: str
    sub_questions: list[SubQuestion]
    fallback: bool = False  # True if extraction failed and single-question fallback was used

    def get_leaves(self) -> list[SubQuestion]:
        return [sq for sq in self.sub_questions if sq.is_leaf]

    def get_hop_count(self) -> int:
        """Longest dependency chain in the DAG (= estimated hop count)."""
        if not self.sub_questions:
            return 0
        id_to_sq = {sq.id: sq for sq in self.sub_questions}
        memo: dict[str, int] = {}

        def depth(sq_id: str) -> int:
            if sq_id in memo:
                return memo[sq_id]
            sq = id_to_sq.get(sq_id)
            if not sq or not sq.depends_on:
                memo[sq_id] = 1
                return 1
            d = 1 + max(depth(dep) for dep in sq.depends_on if dep in id_to_sq)
            memo[sq_id] = d
            return d

        return max(depth(sq.id) for sq in self.sub_questions)

    def get_execution_order(self) -> list[list[SubQuestion]]:
        """Topological sort — returns batches of sub-questions that can run in parallel."""
        id_to_sq = {sq.id: sq for sq in self.sub_questions}
        resolved: set[str] = set()
        remaining = list(self.sub_questions)
        order: list[list[SubQuestion]] = []

        while remaining:
            batch = [sq for sq in remaining if all(d in resolved for d in sq.depends_on)]
            if not batch:
                # Cycle detected — add all remaining as one batch
                logger.warning("DAG cycle detected; flattening remaining nodes")
                order.append(remaining)
                break
            order.append(batch)
            for sq in batch:
                resolved.add(sq.id)
            remaining = [sq for sq in remaining if sq.id not in resolved]

        return order


def _parse_response(text: str) -> list[dict] | None:
    """Extract JSON from LLM response, return sub_questions list or None."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(text)
        sqs = data.get("sub_questions") or data.get("sub_questions\n")
        if isinstance(sqs, list) and sqs:
            return sqs
    except json.JSONDecodeError:
        pass
    return None


def _build_dag(query: str, sqs_raw: list[dict]) -> QuestionDAG:
    dep_ids = {sq["id"] for sq in sqs_raw for _ in [sq.get("depends_on", [])]}
    all_ids = {sq["id"] for sq in sqs_raw}
    sub_questions = [
        SubQuestion(
            id=sq["id"],
            text=sq["text"],
            depends_on=sq.get("depends_on", []),
            is_leaf=not sq.get("depends_on"),
        )
        for sq in sqs_raw
    ]
    return QuestionDAG(root_query=query, sub_questions=sub_questions)


def _fallback_dag(query: str) -> QuestionDAG:
    return QuestionDAG(
        root_query=query,
        sub_questions=[SubQuestion(id="q1", text=query, depends_on=[], is_leaf=True)],
        fallback=True,
    )


def extract_dag(query: str, llm_client: LLMClient) -> QuestionDAG:
    """Decompose a multi-hop query into a dependency DAG of sub-questions."""
    prompts = [_PROMPT.format(query=query), _PROMPT_STRICT.format(query=query)]

    for attempt, prompt in enumerate(prompts):
        response = llm_client.generate(prompt, max_new_tokens=512, temperature=0.0)
        sqs_raw = _parse_response(response.text)
        if sqs_raw:
            return _build_dag(query, sqs_raw)
        logger.warning("DAG parse failed attempt %d for: %s", attempt + 1, query[:80])

    # Both retries failed — log and fall back
    _FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_FAILURE_LOG, "a", encoding="utf-8") as f:
        f.write(f"{query}\n")
    logger.error("DAG extraction failed after 2 attempts; using fallback for: %s", query[:80])
    return _fallback_dag(query)
