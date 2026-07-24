"""Knowledge Library — tennis coaching knowledge indexed by feature name.

Consumers (e.g. the Coaching Engine) should look up entries via
:class:`KnowledgeLibrary` and never hard-code coaching copy.
"""

from src.knowledge_library.library import KnowledgeLibrary, KnowledgeLibraryError
from src.knowledge_library.models import KnowledgeEntry

__all__ = [
    "KnowledgeEntry",
    "KnowledgeLibrary",
    "KnowledgeLibraryError",
]
