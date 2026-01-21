"""Domain-agnostic ground truth data structures.

These classes provide the foundation for evaluation across all biological
domains. They define how queries and expected results are structured,
stored, and loaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GroundTruthQuery:
    """A single ground truth query for evaluation.

    Represents a search query along with the expected entity IDs that
    should be returned. Used to evaluate embedding model and index
    performance.

    Attributes:
        query: The search query text.
        expected: List of entity IDs that should be returned for this query.
        category: Query category (e.g., exact_match, synonym_match, fuzzy_match).
        difficulty: Difficulty level (easy, medium, hard).
        notes: Additional context about the query.
    """

    query: str
    expected: list[str]
    category: str
    difficulty: str = "medium"
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "expected": self.expected,
            "category": self.category,
            "difficulty": self.difficulty,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GroundTruthQuery:
        """Create from dictionary."""
        return cls(
            query=data["query"],
            expected=data["expected"],
            category=data["category"],
            difficulty=data.get("difficulty", "medium"),
            notes=data.get("notes", ""),
        )


@dataclass
class GroundTruthDataset:
    """Collection of ground truth queries for a domain.

    Provides methods for adding queries, filtering by category,
    and serialization to/from JSON.

    Attributes:
        queries: List of ground truth queries.
        metadata: Dataset metadata (source, generation date, domain, etc.).
    """

    queries: list[GroundTruthQuery] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add(self, query: GroundTruthQuery) -> None:
        """Add a query to the dataset."""
        self.queries.append(query)

    def filter_by_category(self, category: str) -> list[GroundTruthQuery]:
        """Get queries of a specific category."""
        return [q for q in self.queries if q.category == category]

    def filter_by_difficulty(self, difficulty: str) -> list[GroundTruthQuery]:
        """Get queries of a specific difficulty."""
        return [q for q in self.queries if q.difficulty == difficulty]

    @property
    def categories(self) -> set[str]:
        """Get all unique categories in the dataset."""
        return {q.category for q in self.queries}

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "metadata": {
                **self.metadata,
                "total_queries": len(self.queries),
                "categories": list(self.categories),
            },
            "queries": [q.to_dict() for q in self.queries],
        }

    def save(self, path: Path | str) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> GroundTruthDataset:
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        dataset = cls(metadata=data.get("metadata", {}))
        for q in data.get("queries", []):
            dataset.add(GroundTruthQuery.from_dict(q))
        return dataset

    def __len__(self) -> int:
        """Return number of queries."""
        return len(self.queries)

    def __iter__(self):
        """Iterate over queries."""
        return iter(self.queries)
