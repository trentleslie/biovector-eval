"""Base entity dataclass for biological entities across domains."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BaseEntity:
    """A biological entity with an identifier, name, and optional synonyms.

    This is the domain-agnostic representation of entities that can be
    embedded and searched. Different domains (metabolites, demographics,
    questionnaires) all convert their specific data formats to this
    common structure.

    Attributes:
        id: Unique identifier for the entity (e.g., HMDB ID, variable code).
        name: Primary name or label for the entity.
        synonyms: Alternative names/labels that should also match this entity.
        metadata: Domain-specific metadata (e.g., source file, category).
    """

    id: str
    name: str
    synonyms: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "synonyms": self.synonyms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BaseEntity:
        """Create from dictionary.

        Supports both standardized format (id, name, synonyms) and
        legacy metabolite format (hmdb_id, name, synonyms).
        """
        # Support legacy metabolite format
        entity_id = data.get("id") or data.get("hmdb_id", "")
        return cls(
            id=entity_id,
            name=data.get("name", ""),
            synonyms=data.get("synonyms", []),
            metadata=data.get("metadata", {}),
        )

    @property
    def all_names(self) -> list[str]:
        """Get primary name plus all synonyms."""
        return [self.name] + self.synonyms

    def __hash__(self) -> int:
        """Hash based on ID for use in sets/dicts."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on ID."""
        if isinstance(other, BaseEntity):
            return self.id == other.id
        return False
