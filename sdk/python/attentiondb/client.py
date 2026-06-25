"""Core client implementation for the AttentionDB REST API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError


class AttentionDBError(Exception):
    """Base exception for AttentionDB API errors."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{status_code}] {message}")


class RateLimitError(AttentionDBError):
    """Raised when the API rate limit is exceeded."""

    def __init__(self, retry_after: int = 1):
        self.retry_after = retry_after
        super().__init__(429, f"Rate limit exceeded. Retry after {retry_after}s")


@dataclass
class SearchResult:
    """A single search result."""
    id: str
    score: float
    fields: Dict[str, str] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Response from a search query."""
    results: List[SearchResult]
    latency_ms: float
    total_count: int
    has_more: bool = False
    offset: int = 0


@dataclass
class BackupInfo:
    """Information about a backup."""
    backup_id: str
    timestamp: str
    collections: List[str]
    path: str
    size_bytes: int


class AttentionDB:
    """AttentionDB REST API client.

    Args:
        base_url: Server URL (e.g. "http://localhost:8080").
        api_key: Optional API key for authentication.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body else None
        req = Request(url, data=data, headers=self._headers(), method=method)

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                content = resp.read().decode("utf-8")
                if content:
                    return json.loads(content)
                return None
        except HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", "1"))
                raise RateLimitError(retry_after) from e
            body = e.read().decode("utf-8", errors="replace")
            raise AttentionDBError(e.code, body) from e

    # ── Collection Management ─────────────────────────────────────────────

    def create_collection(
        self,
        name: str,
        dimension: int = 64,
        heads: Optional[List[str]] = None,
        settings: Optional[Dict[str, Any]] = None,
        head_settings: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create a new collection.

        Args:
            name: Collection name.
            dimension: Vector dimension (max 4096).
            heads: List of head names. Defaults to ["default"].
            settings: HNSW settings override.
            head_settings: Per-head HNSW settings.

        Returns:
            API response with success and message.
        """
        body: Dict[str, Any] = {
            "collection": name,
            "dimension": dimension,
        }
        if heads:
            if not head_settings:
                head_settings = {h: {} for h in heads}
            body["head_settings"] = head_settings
        if settings:
            body["settings"] = settings
        return self._request("POST", "/v1/collections", body)

    # ── Document Operations ───────────────────────────────────────────────

    def insert(
        self,
        collection: str,
        fields: Dict[str, str],
    ) -> str:
        """Insert a document into a collection.

        Args:
            collection: Collection name.
            fields: Document fields. Fields ending in ``_vector``, ``_embedding``,
                    or ``_head`` are parsed as float vectors.

        Returns:
            Document UUID.

        Raises:
            AttentionDBError: If the insertion fails.
        """
        resp = self._request("POST", "/v1/insert", {
            "collection": collection,
            "fields": fields,
        })
        return resp["id"]

    def search(
        self,
        collection: str,
        query: str,
        heads: Optional[List[str]] = None,
        top_k: int = 10,
        min_weight: float = 0.01,
        offset: int = 0,
        hybrid: bool = False,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        query_text: Optional[str] = None,
    ) -> SearchResponse:
        """Search a collection using multi-head attention.

        Args:
            collection: Collection name.
            query: Comma-separated float vector string.
            heads: Heads to search. Defaults to all heads.
            top_k: Number of results (max 10000).
            min_weight: Minimum fusion weight threshold.
            offset: Pagination offset.
            hybrid: Enable BM25 + vector hybrid search.
            bm25_weight: BM25 weight in hybrid fusion.
            vector_weight: Vector weight in hybrid fusion.
            query_text: Text query for BM25 (required if hybrid=True).

        Returns:
            SearchResponse with results and metadata.
        """
        body: Dict[str, Any] = {
            "collection": collection,
            "query": query,
            "top_k": top_k,
            "min_weight": min_weight,
            "offset": offset,
            "hybrid": hybrid,
            "bm25_weight": bm25_weight,
            "vector_weight": vector_weight,
            "query_text": query_text or query,
        }
        if heads:
            body["heads"] = heads

        resp = self._request("POST", "/v1/attend", body)

        return SearchResponse(
            results=[
                SearchResult(
                    id=r["id"],
                    score=r["score"],
                    fields=r.get("fields", {}),
                )
                for r in resp.get("results", [])
            ],
            latency_ms=resp.get("latency_ms", 0.0),
            total_count=resp.get("total_count", 0),
            has_more=resp.get("has_more", False),
            offset=resp.get("offset", 0),
        )

    # ── Health ────────────────────────────────────────────────────────────

    def health(self) -> Dict[str, str]:
        """Get server health status.

        Returns:
            Dict with status and version.
        """
        return self._request("GET", "/health")

    def health_live(self) -> bool:
        """Liveness probe. Returns True if server is alive."""
        try:
            self._request("GET", "/health/live")
            return True
        except AttentionDBError:
            return False

    def health_ready(self) -> bool:
        """Readiness probe. Returns True if server is ready."""
        try:
            self._request("GET", "/health/ready")
            return True
        except AttentionDBError:
            return False

    # ── Admin ─────────────────────────────────────────────────────────────

    def create_backup(self, destination: Optional[str] = None) -> BackupInfo:
        """Create a backup of all collections.

        Args:
            destination: Optional backup path.

        Returns:
            BackupInfo with backup details.
        """
        body = {}
        if destination:
            body["destination"] = destination
        resp = self._request("POST", "/v1/admin/backup", body)
        return BackupInfo(
            backup_id=resp["backup_id"],
            timestamp=resp["timestamp"],
            collections=resp.get("collections", []),
            path=resp["path"],
            size_bytes=resp.get("size_bytes", 0),
        )

    def list_backups(self) -> List[BackupInfo]:
        """List all available backups.

        Returns:
            List of BackupInfo objects.
        """
        resp = self._request("GET", "/v1/admin/backups")
        return [
            BackupInfo(
                backup_id=b["backup_id"],
                timestamp=b["timestamp"],
                collections=b.get("collections", []),
                path=b["path"],
                size_bytes=b.get("size_bytes", 0),
            )
            for b in resp.get("backups", [])
        ]
