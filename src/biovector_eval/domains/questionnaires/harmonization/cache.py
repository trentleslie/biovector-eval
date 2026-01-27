"""Caching layer for GenOMA results.

Provides persistent caching to avoid repeated LLM calls for the same
input text. Cache keys include the model version to enable cache
invalidation when the underlying model changes.

Typical GenOMA cost structure:
- First run (500 questions): ~$15-30, 20-40 min
- Subsequent runs: ~$0, instant (cache hits)
- Incremental updates: Only new questions hit API
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class GenOMACache:
    """Cache GenOMA results to avoid repeated LLM calls.

    Uses content-addressable storage based on SHA256 hash of:
    - Input text
    - Field type
    - Model version (for cache invalidation)

    Example:
        cache = GenOMACache()
        result = cache.get("How often do you feel anxious?", "radio")
        if result is None:
            result = genoma_graph.invoke(...)
            cache.set("How often do you feel anxious?", "radio", result)
    """

    def __init__(
        self,
        cache_dir: Path | str = "data/cache/genoma",
        model_version: str = "gpt-4",
    ):
        """Initialize cache.

        Args:
            cache_dir: Directory to store cache files
            model_version: Model version string for cache key (invalidates
                          on model change)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_version = model_version
        self._hits = 0
        self._misses = 0

    def _hash_key(self, text: str, field_type: str) -> str:
        """Create cache key from input including model version for invalidation.

        Args:
            text: Input text to hash
            field_type: GenOMA field type (radio, checkbox, short)

        Returns:
            16-character hex hash string
        """
        content = f"{text}|{field_type}|{self.model_version}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"{key}.json"

    def get(self, text: str, field_type: str) -> dict[str, Any] | None:
        """Retrieve cached result if exists.

        Args:
            text: Input text that was mapped
            field_type: Field type used for mapping

        Returns:
            Cached result dict, or None if not in cache
        """
        key = self._hash_key(text, field_type)
        cache_file = self._cache_path(key)
        if cache_file.exists():
            self._hits += 1
            return json.loads(cache_file.read_text(encoding="utf-8"))
        self._misses += 1
        return None

    def set(self, text: str, field_type: str, result: dict[str, Any]) -> None:
        """Cache a GenOMA result.

        Args:
            text: Input text that was mapped
            field_type: Field type used for mapping
            result: GenOMA result dict to cache
        """
        key = self._hash_key(text, field_type)
        cache_file = self._cache_path(key)
        cache_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def has(self, text: str, field_type: str) -> bool:
        """Check if a result is cached without loading it.

        Args:
            text: Input text to check
            field_type: Field type to check

        Returns:
            True if result is cached
        """
        key = self._hash_key(text, field_type)
        return self._cache_path(key).exists()

    def delete(self, text: str, field_type: str) -> bool:
        """Delete a cached result.

        Args:
            text: Input text whose result to delete
            field_type: Field type of result to delete

        Returns:
            True if cache entry was deleted, False if not found
        """
        key = self._hash_key(text, field_type)
        cache_file = self._cache_path(key)
        if cache_file.exists():
            cache_file.unlink()
            return True
        return False

    def clear(self) -> int:
        """Clear all cached results.

        Returns:
            Number of cache entries deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count

    @property
    def stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, hit_rate, and total cached entries
        """
        cached_count = len(list(self.cache_dir.glob("*.json")))
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
            "cached_entries": cached_count,
        }

    def list_cached(self) -> list[str]:
        """List all cache keys.

        Returns:
            List of cache key strings (truncated hashes)
        """
        return [f.stem for f in self.cache_dir.glob("*.json")]
