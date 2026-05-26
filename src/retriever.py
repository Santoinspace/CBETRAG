"""Retriever abstraction layer — wraps ElasticSearch and in-memory sources."""
from __future__ import annotations
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return list of passage strings ranked by relevance."""
        ...


class ElasticRetriever(Retriever):
    """BM25 retrieval against an ElasticSearch Wikipedia index.

    Uses the raw elasticsearch-py client for reliability (BEIR wrapper
    may not return full document sources across versions).
    """

    def __init__(self, index_name: str = "wiki", host: str = "localhost",
                 port: int = 9200, top_k: int = 5):
        from elasticsearch import Elasticsearch
        self._es = Elasticsearch([{"host": host, "port": port}])
        self._index = index_name
        self._top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        k = top_k or self._top_k
        try:
            body = {
                "query": {
                    "match": {"txt": query}
                },
                "size": k,
                "_source": ["txt"],
            }
            resp = self._es.search(index=self._index, body=body)
            hits = resp["hits"]["hits"]
            return [h["_source"].get("txt", "") for h in hits]
        except Exception as e:
            logger.warning("ES retrieve failed for '%s': %s", query[:60], e)
            return []


class PassageListRetriever(Retriever):
    """In-memory retriever for testing — returns passages from a fixed list."""

    def __init__(self, passages: list[str]):
        self._passages = passages

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        return self._passages[:top_k]
