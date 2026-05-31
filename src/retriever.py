"""Retriever abstraction layer — wraps ElasticSearch and in-memory sources."""
from __future__ import annotations
from abc import ABC, abstractmethod
from functools import lru_cache
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

    Includes LRU cache (10k entries) to avoid repeated queries for the
    same search string across samples.
    """

    def __init__(self, index_name: str = "wiki", host: str = "localhost",
                 port: int = 9200, top_k: int = 5):
        from elasticsearch import Elasticsearch
        self._es = Elasticsearch([{"host": host, "port": port}])
        self._index = index_name
        self._top_k = top_k
        self._cache_hits = 0
        self._cache_misses = 0

    def is_available(self) -> bool:
        """Check if ES cluster is reachable."""
        try:
            return self._es.ping()
        except Exception:
            return False

    def get_doc_count(self) -> int:
        """Return total number of documents in the index."""
        try:
            resp = self._es.count(index=self._index)
            return resp.get("count", 0)
        except Exception as e:
            logger.warning("get_doc_count failed: %s", e)
            return 0

    @lru_cache(maxsize=10000)
    def _retrieve_cached(self, query: str, top_k: int) -> tuple:
        """ES retrieval with LRU cache. Same query → same results."""
        try:
            body = {
                "query": {
                    "match": {"txt": query}
                },
                "size": top_k,
                "_source": ["txt"],
            }
            resp = self._es.search(index=self._index, body=body)
            hits = resp["hits"]["hits"]
            return tuple(h["_source"].get("txt", "") for h in hits)
        except Exception as e:
            logger.warning("ES retrieve failed for '%s': %s", query[:60], e)
            return ()

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        k = top_k or self._top_k
        return list(self._retrieve_cached(query, k))

    def cache_stats(self) -> str:
        info = self._retrieve_cached.cache_info()
        return f"ES cache: hits={info.hits} misses={info.misses} size={info.currsize}/{info.maxsize}"


class PassageListRetriever(Retriever):
    """In-memory retriever for testing — returns passages from a fixed list."""

    def __init__(self, passages: list[str]):
        self._passages = passages

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        return self._passages[:top_k]
