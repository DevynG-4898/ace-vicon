"""Knowledge Library — indexed tennis coaching knowledge for feature lookup."""

from __future__ import annotations

from src.knowledge_library.entries import DEFAULT_ENTRIES
from src.knowledge_library.models import KnowledgeEntry


class KnowledgeLibraryError(LookupError):
    """Raised when a knowledge lookup fails."""


class KnowledgeLibrary:
    """Read-only library indexed by feature name.

    The Coaching Engine depends only on this lookup API; it contains no
    tennis-specific strings of its own.
    """

    def __init__(self, entries: tuple[KnowledgeEntry, ...] | None = None) -> None:
        source = DEFAULT_ENTRIES if entries is None else entries
        index: dict[str, KnowledgeEntry] = {}
        for entry in source:
            if entry.feature in index:
                raise ValueError(f"duplicate knowledge entry for feature: {entry.feature!r}")
            index[entry.feature] = entry
        self._entries = index

    @classmethod
    def default(cls) -> KnowledgeLibrary:
        """Return a library loaded with the built-in default entries."""
        return cls()

    def get(self, feature: str) -> KnowledgeEntry:
        """Look up a knowledge entry by exact feature name."""
        try:
            return self._entries[feature]
        except KeyError as exc:
            known = ", ".join(sorted(self._entries)) or "(empty)"
            raise KnowledgeLibraryError(
                f"No knowledge entry for feature {feature!r}. "
                f"Known features: {known}"
            ) from exc

    def __contains__(self, feature: str) -> bool:
        return feature in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def features(self) -> tuple[str, ...]:
        """Sorted feature names present in the library."""
        return tuple(sorted(self._entries))
