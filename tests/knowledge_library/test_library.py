"""Unit tests for the Knowledge Library."""

from __future__ import annotations

import pytest

from src.knowledge_library import KnowledgeEntry, KnowledgeLibrary, KnowledgeLibraryError


def test_default_library_contains_right_elbow_flexion():
    library = KnowledgeLibrary.default()
    entry = library.get("Right Elbow Flexion")
    assert entry.phase == "Cocking"
    assert "Bend your right elbow more" in entry.too_low
    assert "Reduce right elbow bend" in entry.too_high


def test_feature_lookup_works():
    library = KnowledgeLibrary.default()
    assert "Contact Height" in library
    entry = library.get("Contact Height")
    assert entry.feature == "Contact Height"
    assert entry.phase == "Contact"


def test_missing_feature_produces_clear_error():
    library = KnowledgeLibrary.default()
    with pytest.raises(KnowledgeLibraryError, match="No knowledge entry for feature 'Not A Feature'"):
        library.get("Not A Feature")


def test_too_low_and_too_high_corrections():
    entry = KnowledgeLibrary.default().get("Right Elbow Flexion")
    assert entry.correction_for("too_low") == entry.too_low
    assert entry.correction_for("too_high") == entry.too_high


def test_coach_quotes_returned():
    entry = KnowledgeLibrary.default().get("Right Elbow Flexion")
    assert "Let the elbow fold naturally." in entry.coach_quotes
    assert len(entry.coach_quotes) == 3


def test_practice_drills_returned():
    entry = KnowledgeLibrary.default().get("Right Elbow Flexion")
    assert "Serve without a ball." in entry.practice_drills
    assert "Hold the trophy position before accelerating." in entry.practice_drills


def test_custom_library_rejects_duplicates():
    entries = (
        KnowledgeEntry(feature="A", phase="Loading", too_low="x", too_high="y"),
        KnowledgeEntry(feature="A", phase="Loading", too_low="x2", too_high="y2"),
    )
    with pytest.raises(ValueError, match="duplicate"):
        KnowledgeLibrary(entries)


def test_empty_quotes_and_drills_are_tuples():
    entry = KnowledgeLibrary.default().get("Knee Flexion")
    assert entry.coach_quotes == ()
    assert entry.practice_drills == ()
