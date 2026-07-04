import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import db
from core.prompts import (
    build_events_and_experiences_prompt,
    build_hidden_gems_prompt,
    build_recommendation_prompt,
    build_storytelling_prompt,
    _language_instruction,
    _profile_summary,
)
from core.retrieval import _keyword_score, _record_text, load_dataset


# ---------- retrieval tests ----------

def test_load_destinations_dataset_has_expected_shape():
    records = load_dataset("destinations.json")
    assert len(records) > 5
    for r in records:
        assert "name" in r and "state" in r and "description" in r


def test_load_experiences_dataset_has_expected_shape():
    records = load_dataset("experiences.json")
    assert len(records) > 3
    for r in records:
        assert "title" in r and "location" in r


def test_record_text_includes_name_and_tags():
    record = {"name": "Hampi", "state": "Karnataka", "description": "Ancient ruins",
              "type": ["heritage"], "tags": ["UNESCO", "offbeat"]}
    text = _record_text(record)
    assert "Hampi" in text
    assert "UNESCO" in text
    assert "heritage" in text


def test_keyword_score_matches_relevant_terms():
    text = "hampi karnataka ancient ruins temples unesco offbeat"
    score_high = _keyword_score("ancient temples ruins", text)
    score_low = _keyword_score("beach goa nightlife", text)
    assert score_high > score_low


def test_keyword_score_empty_query_returns_zero():
    assert _keyword_score("", "some text") == 0.0


# ---------- prompt building tests ----------

def test_profile_summary_includes_provided_fields():
    profile = {"name": "Priya", "age_band": "26-40", "location": "Mumbai", "interests": ["Food & cuisine"]}
    summary = _profile_summary(profile)
    assert "Priya" in summary
    assert "Mumbai" in summary
    assert "Food & cuisine" in summary


def test_profile_summary_handles_empty_profile():
    summary = _profile_summary({})
    assert "No profile details" in summary


def test_language_instruction_variants():
    assert "Hindi" in _language_instruction({"language_pref": "Hindi"})
    assert "English" in _language_instruction({"language_pref": "English"})
    both = _language_instruction({"language_pref": "English + Hindi"})
    assert "English" in both and "Hindi" in both


def test_recommendation_prompt_includes_context_and_query():
    retrieved = [{"name": "Hampi", "state": "Karnataka", "region": "South India",
                  "description": "Ruins", "best_season": "Winter"}]
    system, user = build_recommendation_prompt("heritage trip", {"name": "Ravi"}, retrieved)
    assert "Ravi" in user
    assert "heritage trip" in user
    assert "Hampi" in user
    assert isinstance(system, str) and len(system) > 0


def test_hidden_gems_prompt_mentions_location():
    system, user = build_hidden_gems_prompt("Coorg", {"name": "Anu"})
    assert "Coorg" in user
    assert "Anu" in user


def test_storytelling_prompt_mentions_place():
    system, user = build_storytelling_prompt("Varanasi", {"name": "Sam"})
    assert "Varanasi" in user


def test_events_prompt_includes_curated_experiences():
    experiences = [{"title": "Pottery workshop", "location": "Khurja", "category": "craft",
                     "description": "Hands-on pottery"}]
    system, user = build_events_and_experiences_prompt("Khurja", {"name": "Dev"}, experiences)
    assert "Pottery workshop" in user
    assert "Khurja" in user


# ---------- db tests ----------

@pytest.fixture()
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db.init_db(db_path=path)
    yield path
    os.remove(path)


def test_save_and_get_profile(temp_db):
    profile = {"name": "Test User", "age_band": "26-40", "gender": "Female",
               "location": "Delhi", "companions": "Solo", "interests": ["Nature & wildlife"],
               "budget": "Mid-range", "language_pref": "English"}
    db.save_profile("test@example.com", profile, db_path=temp_db)
    fetched = db.get_profile("test@example.com", db_path=temp_db)
    assert fetched is not None
    assert fetched["name"] == "Test User"
    assert fetched["interests"] == ["Nature & wildlife"]


def test_profile_identifier_is_case_insensitive(temp_db):
    db.save_profile("Test@Example.com", {"name": "X"}, db_path=temp_db)
    assert db.get_profile("test@example.com", db_path=temp_db) is not None


def test_get_profile_returns_none_when_missing(temp_db):
    assert db.get_profile("nobody@example.com", db_path=temp_db) is None


def test_add_and_get_history(temp_db):
    db.save_profile("hist@example.com", {"name": "H"}, db_path=temp_db)
    db.add_history_entry("hist@example.com", "recommendation", "heritage trip", "Some AI output", db_path=temp_db)
    history = db.get_history("hist@example.com", db_path=temp_db)
    assert len(history) == 1
    assert history[0]["feature_type"] == "recommendation"
    assert history[0]["query_input"] == "heritage trip"


def test_history_respects_limit(temp_db):
    db.save_profile("many@example.com", {"name": "M"}, db_path=temp_db)
    for i in range(5):
        db.add_history_entry("many@example.com", "recommendation", f"query {i}", "output", db_path=temp_db)
    history = db.get_history("many@example.com", limit=3, db_path=temp_db)
    assert len(history) == 3
