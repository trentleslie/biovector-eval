"""Domain registry for biological entity evaluation.

This module provides a central registry for accessing different biological
domains (metabolites, demographics, questionnaires). Each domain implements
the Domain protocol and can be retrieved by name.

Usage:
    from biovector_eval.domains import get_domain, list_domains

    # Get a specific domain
    metabolites = get_domain("metabolites")

    # List available domains
    available = list_domains()  # ["metabolites", "demographics", "questionnaires"]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biovector_eval.base.domain import Domain

# Domain registry - populated by importing domain modules
_DOMAIN_REGISTRY: dict[str, type[Domain]] = {}


def register_domain(name: str, domain_class: type[Domain]) -> None:
    """Register a domain class.

    Args:
        name: Domain identifier (e.g., "metabolites").
        domain_class: Class implementing Domain protocol.
    """
    _DOMAIN_REGISTRY[name] = domain_class


def get_domain(name: str, **kwargs) -> Domain:
    """Get a domain instance by name.

    Args:
        name: Domain identifier.
        **kwargs: Arguments passed to domain constructor.

    Returns:
        Domain instance.

    Raises:
        KeyError: If domain name is not registered.
    """
    if name not in _DOMAIN_REGISTRY:
        raise KeyError(
            f"Unknown domain: {name}. " f"Available: {list(_DOMAIN_REGISTRY.keys())}"
        )
    return _DOMAIN_REGISTRY[name](**kwargs)


def list_domains() -> list[str]:
    """List all registered domain names.

    Returns:
        List of domain identifiers.
    """
    return list(_DOMAIN_REGISTRY.keys())


# Import domain modules to trigger registration
# These imports must be at the end to avoid circular imports
from biovector_eval.domains import metabolites  # noqa: E402, F401

# Future domains - scaffolded but not yet implemented
# from biovector_eval.domains import demographics  # noqa: E402, F401
# from biovector_eval.domains import questionnaires  # noqa: E402, F401


__all__ = [
    "get_domain",
    "list_domains",
    "register_domain",
]
