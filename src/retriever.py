"""Retriever abstraction layer — wraps AdaRAGUE's ElasticSearch index."""
from __future__ import annotations
from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return list of passage strings ranked by relevance."""
        ...


class ElasticRetriever(Retriever):
    """Thin wrapper around AdaRAGUE's BEIR ElasticSearch index."""

    def __init__(self, index_name: str, host: str = "localhost",
                 port: int = 9200, top_k: int = 5):
        from beir.retrieval.search.lexical.elastic_search import ElasticSearch
        config = {
            "hostname": {"host": host, "port": port},
            "index_name": index_name,
            "keys": {"title": "title", "body": "txt"},
            "timeout": 100,
            "retry_on_timeout": True,
            "maxsize": 24,
            "number_of_shards": "default",
            "language": "english",
        }
        self._es = ElasticSearch(config)
        self._top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        k = top_k or self._top_k
        results = self._es.lexical_multisearch(
            texts=[query], top_hits=k
        )
        hits = results[0].get("hits", {}).get("hits", [])
        return [h["_source"].get("txt", "") for h in hits]


class PassageListRetriever(Retriever):
    """In-memory retriever for testing — returns passages from a fixed list."""

    def __init__(self, passages: list[str]):
        self._passages = passages

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        return self._passages[:top_k]
