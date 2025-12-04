"""Ground truth generation for HMDB metabolite evaluation."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class GroundTruthQuery:
    """A single ground truth query."""

    query: str
    expected: list[str]
    category: str
    difficulty: str = "medium"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "expected": self.expected,
            "category": self.category,
            "difficulty": self.difficulty,
            "notes": self.notes,
        }


@dataclass
class GroundTruthDataset:
    """Collection of ground truth queries."""

    queries: list[GroundTruthQuery] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add(self, query: GroundTruthQuery) -> None:
        """Add a query to the dataset."""
        self.queries.append(query)

    def filter_by_category(self, category: str) -> list[GroundTruthQuery]:
        """Get queries of a specific category."""
        return [q for q in self.queries if q.category == category]

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "metadata": {
                **self.metadata,
                "total_queries": len(self.queries),
                "categories": list({q.category for q in self.queries}),
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
            dataset.add(GroundTruthQuery(**q))
        return dataset


class HMDBGroundTruthGenerator:
    """Generate ground truth queries from HMDB metabolite data."""

    GREEK_MAPPINGS = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ω": "omega",
    }

    def __init__(self, metabolites: list[dict], seed: int = 42):
        """Initialize with metabolite data.

        Args:
            metabolites: List of dicts with 'hmdb_id', 'name', 'synonyms' keys.
            seed: Random seed for reproducibility.
        """
        self.metabolites = metabolites
        self.rng = random.Random(seed)

    def generate_exact_matches(self, n: int = 150) -> Iterator[GroundTruthQuery]:
        """Generate exact name match queries."""
        sampled = self.rng.sample(self.metabolites, min(n, len(self.metabolites)))
        for m in sampled:
            yield GroundTruthQuery(
                query=m["name"],
                expected=[m["hmdb_id"]],
                category="exact_match",
                difficulty="easy",
            )

    def generate_synonym_matches(self, n: int = 150) -> Iterator[GroundTruthQuery]:
        """Generate synonym match queries."""
        with_synonyms = [m for m in self.metabolites if m.get("synonyms")]
        sampled = self.rng.sample(with_synonyms, min(n, len(with_synonyms)))
        for m in sampled:
            synonym = self.rng.choice(m["synonyms"])
            yield GroundTruthQuery(
                query=synonym,
                expected=[m["hmdb_id"]],
                category="synonym_match",
                difficulty="medium",
                notes=f"Synonym for '{m['name']}'",
            )

    def generate_fuzzy_matches(self, n: int = 100) -> Iterator[GroundTruthQuery]:
        """Generate partial/fuzzy match queries."""
        long_names = [m for m in self.metabolites if len(m["name"]) > 8]
        sampled = self.rng.sample(long_names, min(n, len(long_names)))
        for m in sampled:
            prefix_len = self.rng.randint(4, min(7, len(m["name"]) - 1))
            prefix = m["name"][:prefix_len]
            yield GroundTruthQuery(
                query=prefix,
                expected=[m["hmdb_id"]],
                category="fuzzy_match",
                difficulty="hard",
                notes=f"Prefix of '{m['name']}'",
            )

    def generate_edge_cases(self, n: int = 100) -> Iterator[GroundTruthQuery]:
        """Generate edge case queries (Greek letters, special chars)."""
        edge_cases: list[GroundTruthQuery] = []

        # Greek letters
        for m in self.metabolites:
            name_lower = m["name"].lower()
            for greek, english in self.GREEK_MAPPINGS.items():
                if greek in name_lower or english in name_lower:
                    edge_cases.append(
                        GroundTruthQuery(
                            query=m["name"],
                            expected=[m["hmdb_id"]],
                            category="greek_letter",
                            difficulty="hard",
                        )
                    )
                    break

        # Numeric prefixes (e.g., "3-hydroxybutyrate")
        numeric_pattern = re.compile(r"^\d+-")
        for m in self.metabolites:
            if numeric_pattern.match(m["name"]):
                edge_cases.append(
                    GroundTruthQuery(
                        query=m["name"],
                        expected=[m["hmdb_id"]],
                        category="numeric_prefix",
                        difficulty="medium",
                    )
                )

        # Special prefixes (N-acetyl, L-carnitine)
        special_pattern = re.compile(r"^[A-Z]-[a-z]")
        for m in self.metabolites:
            if special_pattern.match(m["name"]):
                edge_cases.append(
                    GroundTruthQuery(
                        query=m["name"],
                        expected=[m["hmdb_id"]],
                        category="special_prefix",
                        difficulty="medium",
                    )
                )

        self.rng.shuffle(edge_cases)
        yield from edge_cases[:n]

    def generate_dataset(
        self,
        exact: int = 150,
        synonym: int = 150,
        fuzzy: int = 100,
        edge: int = 100,
    ) -> GroundTruthDataset:
        """Generate complete ground truth dataset.

        Args:
            exact: Number of exact match queries.
            synonym: Number of synonym match queries.
            fuzzy: Number of fuzzy match queries.
            edge: Maximum number of edge case queries.

        Returns:
            GroundTruthDataset with all queries.
        """
        dataset = GroundTruthDataset(
            metadata={
                "source": "HMDB",
                "generated": datetime.now().isoformat(),
                "version": "1.0",
            }
        )

        for q in self.generate_exact_matches(exact):
            dataset.add(q)
        for q in self.generate_synonym_matches(synonym):
            dataset.add(q)
        for q in self.generate_fuzzy_matches(fuzzy):
            dataset.add(q)
        for q in self.generate_edge_cases(edge):
            dataset.add(q)

        return dataset
