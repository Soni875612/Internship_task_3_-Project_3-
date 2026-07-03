"""
DecodeLabs - Project 3 (Advanced)
Unit tests for the TechStackRecommender engine.

Run:
    pytest tests/ -v
"""

import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recommender import TechStackRecommender, MIN_INPUT_SKILLS  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    return TechStackRecommender()


def test_dataset_loads(engine):
    assert len(engine.df) > 0
    assert {"role", "category", "skills"}.issubset(engine.df.columns)


def test_minimum_skill_guard(engine):
    with pytest.raises(ValueError):
        engine.recommend(["Python", "SQL"])  # only 2, below MIN_INPUT_SKILLS


def test_cold_start_fallback(engine):
    result = engine.recommend([], top_n=3)
    assert len(result) == 3
    assert "note" in result.columns


def test_recommend_returns_top_n(engine):
    result = engine.recommend(["Python", "SQL", "Machine Learning"], top_n=5)
    assert len(result) == 5
    assert list(result.columns).__contains__("match_percent")


def test_recommend_sorted_descending(engine):
    result = engine.recommend(["Python", "SQL", "Machine Learning"], top_n=6)
    scores = result["match_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_exact_match_scores_highest(engine):
    """A user whose skills exactly match a role's should get that role as #1."""
    role_row = engine.df.iloc[0]
    skills = [s.strip() for s in role_row["skills"].split(",")]
    result = engine.recommend(skills, top_n=1)
    assert result.iloc[0]["role"] == role_row["role"]


def test_fuzzy_correction_handles_typos(engine):
    corrections = engine.correct_skills(["Pyhton", "Dokcer"])
    corrected = [c[1] for c in corrections]
    assert "Python" in corrected
    assert "Docker" in corrected


def test_explain_breakdown_matches_skills(engine):
    top_role = engine.recommend(["Python", "SQL", "Machine Learning"], top_n=1).iloc[0]["role"]
    breakdown = engine.explain(["Python", "SQL", "Machine Learning"], top_role)
    assert set(breakdown["skill"]) == {"Python", "SQL", "Machine Learning"}


def test_skill_gap_readiness_between_0_and_100(engine):
    gap = engine.skill_gap(["Python", "SQL", "Machine Learning"], "Data Scientist")
    assert 0 <= gap["readiness_percent"] <= 100
    assert isinstance(gap["skills_missing"], list)


def test_skill_gap_full_match_is_100_percent(engine):
    role_row = engine.df[engine.df["role"] == "Data Scientist"].iloc[0]
    skills = [s.strip() for s in role_row["skills"].split(",")]
    gap = engine.skill_gap(skills, "Data Scientist")
    assert gap["readiness_percent"] == 100.0
    assert gap["skills_missing"] == []


def test_score_all_algorithms_returns_three_columns(engine):
    result = engine.score_all_algorithms(["Python", "SQL", "Machine Learning"])
    assert {"cosine", "jaccard", "euclidean_score"}.issubset(result.columns)


def test_invalid_input_type_raises(engine):
    with pytest.raises(TypeError):
        engine.recommend("Python, SQL, ML")  # string, not list
