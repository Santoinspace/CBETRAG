"""BM25 retriever over dataset passages — mini-index for iteration-retrieval testing.

Builds a BM25 index from all passages across dataset questions (~74k passages),
enabling varied retrieval results per query without requiring a full Wikipedia index.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, index_path: str | None = None):
        self.bm25: BM25Okapi | None = None
        self.passages: list[str] = []
        self.passage_indices: list[int] = []  # question index per passage
        if index_path and Path(index_path).exists():
            self._load(index_path)

    def build_from_dataset(self, questions: list, index_save_path: str | None = None):
        """Build BM25 index from a list of Question objects.

        Extracts all gold + distractor passages, deduplicates, and indexes them.
        """
        seen = set()
        passages = []
        for q in questions:
            for p in q.gold_passages + q.distractor_passages:
                key = p[:100]  # deduplicate by first 100 chars
                if key not in seen:
                    seen.add(key)
                    passages.append(p)

        self.passages = passages
        tokenized = [p.lower().split() for p in passages]
        self.bm25 = BM25Okapi(tokenized)
        if index_save_path:
            self._save(index_save_path)

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        if self.bm25 is None:
            return []
        tokens = query.lower().split()
        if not tokens:
            return self.passages[:top_k]
        scores = self.bm25.get_scores(tokens)
        # Convert to numpy for argsort
        import numpy as np
        top_indices = np.array(scores).argsort()[-top_k:][::-1]
        return [self.passages[i] for i in top_indices]

    def _save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {"passages": self.passages}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.passages = data["passages"]
        tokenized = [p.lower().split() for p in self.passages]
        self.bm25 = BM25Okapi(tokenized)
