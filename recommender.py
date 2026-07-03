"""
DecodeLabs - Project 3 (Advanced): AI Recommendation Logic (Tech Stack Recommender)
====================================================================================
Content-Based Filtering engine built on the IPO (Input -> Process -> Output)
architecture, using TF-IDF vector mapping and Cosine Similarity.

ADVANCED FEATURES (v2):
    - Fuzzy skill correction: typos / near-matches ("Pyhton", "ReactJS") are
      auto-corrected to the nearest known vocabulary skill before scoring,
      directly solving the "vocabulary mismatch" problem described in the
      training material ("Web Design" vs "Frontend Development").
    - Explainability: `explain()` breaks down exactly which of the user's
      skills contributed to a role's match score, and by how much.
    - Skill-Gap Analysis: `skill_gap()` reports which of a role's required
      skills the user is still missing — turning the engine into a career
      roadmap tool, not just a ranking table.
    - Multi-algorithm scoring: Cosine Similarity, Jaccard Similarity, and
      Euclidean-based scoring are all available for benchmarking via
      `score_all_algorithms()`.

Pipeline:
    1. Ingestion   -> capture + fuzzy-correct user skills (min. 3 required)
    2. Scoring     -> TF-IDF weighting + Cosine Similarity vs. job roles
    3. Sorting     -> rank roles by similarity score (descending)
    4. Filtering   -> truncate to Top-N to avoid choice overload

Cold-start handling:
    - User Cold Start -> Trending Fallback (most common/popular roles)
    - Item Cold Start -> not applicable here (content-based, works day 1)
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from rapidfuzz import process as fuzzy_process, fuzz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw_skills.csv")

MIN_INPUT_SKILLS = 3
FUZZY_MATCH_THRESHOLD = 75  # 0-100; below this, the raw skill is kept as-is


class TechStackRecommender:
    """
    A content-based recommendation engine that maps a user's raw skills
    to the most relevant tech job roles using TF-IDF + Cosine Similarity,
    with fuzzy correction, explainability, and skill-gap analysis.
    """

    def __init__(self, data_path: str = DATA_PATH):
        self.data_path = data_path
        self.df = None
        self.vectorizer = None
        self.role_matrix = None
        self.known_skills = []
        self._fit()

    # ------------------------------------------------------------------
    # STEP 0 — INGESTION (data load) + PROCESS (vector mapping / TF-IDF)
    # ------------------------------------------------------------------
    def _fit(self):
        """Load the dataset and build the TF-IDF vector space (shared vocabulary)."""
        self.df = pd.read_csv(self.data_path)
        self.df["skills_list"] = self.df["skills"].apply(
            lambda s: [x.strip() for x in s.split(",") if x.strip()]
        )
        self.df["skills_clean"] = self.df["skills"].apply(self._normalize)

        # Word-level TF-IDF: multi-word skills ("Cloud Computing") are split
        # into unigrams ("cloud", "computing") so related skills sharing a
        # root word still overlap in vector space.
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\+\#\.]*\b")
        self.role_matrix = self.vectorizer.fit_transform(self.df["skills_clean"])

        # Canonical skill vocabulary (title-case, deduplicated) used as the
        # fuzzy-matching correction target.
        seen = {}
        for s in self.df["skills_list"].explode():
            seen[s.strip().lower()] = s.strip()
        self.known_skills = list(seen.values())

    @staticmethod
    def _normalize(raw_skills) -> str:
        """Turn a comma-separated skill string (or list) into a clean,
        lowercase, space-separated word list ready for TF-IDF tokenization."""
        if isinstance(raw_skills, list):
            tokens = raw_skills
        else:
            tokens = raw_skills.split(",")
        cleaned = [t.strip().lower() for t in tokens if t.strip()]
        return " ".join(cleaned)

    # ------------------------------------------------------------------
    # ADVANCED — Fuzzy skill correction (handles typos / naming mismatches)
    # ------------------------------------------------------------------
    def correct_skills(self, user_skills):
        """
        Map each raw user skill to its closest known vocabulary skill using
        fuzzy string matching. Returns a list of (original, corrected, score)
        tuples so the correction is fully transparent to the caller.
        """
        corrections = []
        for raw in user_skills:
            match = fuzzy_process.extractOne(
                raw, self.known_skills, scorer=fuzz.WRatio
            )
            if match and match[1] >= FUZZY_MATCH_THRESHOLD:
                corrections.append((raw, match[0], round(match[1], 1)))
            else:
                corrections.append((raw, raw, 0.0))  # no confident match, keep as-is
        return corrections

    def popular_roles(self, top_n: int = 3) -> pd.DataFrame:
        """Trending Fallback for the User Cold Start problem."""
        return self.df[["role", "category", "skills"]].head(top_n)

    # ------------------------------------------------------------------
    # STEP 1 — SCORING  (Cosine Similarity)
    # STEP 2 — SORTING  (descending rank)
    # STEP 3 — FILTERING (Top-N cutoff)
    # ------------------------------------------------------------------
    def recommend(self, user_skills, top_n: int = 3, use_fuzzy: bool = True) -> pd.DataFrame:
        """
        Take a list of raw user skills and return the Top-N most similar
        job roles ranked by Cosine Similarity score.
        """
        if not isinstance(user_skills, (list, tuple)):
            raise TypeError("user_skills must be a list of strings")

        if len(user_skills) == 0:
            fallback = self.popular_roles(top_n).copy()
            fallback["match_score"] = 0.0
            fallback["match_percent"] = 0.0
            fallback["note"] = "Trending fallback (cold start — no skills provided)"
            return fallback

        if len(user_skills) < MIN_INPUT_SKILLS:
            raise ValueError(
                f"Please provide at least {MIN_INPUT_SKILLS} skills for accurate matching."
            )

        effective_skills = user_skills
        if use_fuzzy:
            corrections = self.correct_skills(user_skills)
            effective_skills = [c[1] for c in corrections]

        user_vector_text = self._normalize(",".join(effective_skills))
        user_vector = self.vectorizer.transform([user_vector_text])

        scores = cosine_similarity(user_vector, self.role_matrix).flatten()

        result = self.df[["role", "category", "skills"]].copy()
        result["match_score"] = np.round(scores, 4)
        result = result.sort_values("match_score", ascending=False)

        top_result = result.head(top_n).reset_index(drop=True)
        top_result["match_percent"] = (top_result["match_score"] * 100).round(1)

        return top_result

    # ------------------------------------------------------------------
    # ADVANCED — Explainability
    # ------------------------------------------------------------------
    def explain(self, user_skills, role_name: str, use_fuzzy: bool = True) -> pd.DataFrame:
        """
        Break down exactly which of the user's (corrected) skills overlap
        with a given role's TF-IDF vocabulary, and how much each one
        contributes to the final cosine similarity score.
        """
        effective_skills = user_skills
        if use_fuzzy:
            effective_skills = [c[1] for c in self.correct_skills(user_skills)]

        role_idx = self.df.index[self.df["role"] == role_name][0]
        role_vec = self.role_matrix[role_idx].toarray().flatten()
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        rows = []
        for skill in effective_skills:
            skill_tokens = self._normalize([skill]).split()
            contribution = 0.0
            for tok in skill_tokens:
                if tok in feature_names:
                    idx = np.where(feature_names == tok)[0][0]
                    contribution += role_vec[idx]
            rows.append({
                "skill": skill,
                "present_in_role": contribution > 0,
                "contribution_weight": round(float(contribution), 4),
            })

        out = pd.DataFrame(rows).sort_values("contribution_weight", ascending=False)
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # ADVANCED — Skill-Gap Analysis (career roadmap)
    # ------------------------------------------------------------------
    def skill_gap(self, user_skills, role_name: str, use_fuzzy: bool = True) -> dict:
        """
        Compare the user's (corrected) skills against a role's required
        skill set and report what's already covered vs. what's missing —
        effectively a mini learning roadmap.
        """
        effective_skills = user_skills
        if use_fuzzy:
            effective_skills = [c[1] for c in self.correct_skills(user_skills)]

        user_set = {s.strip().lower() for s in effective_skills}
        role_row = self.df[self.df["role"] == role_name].iloc[0]
        required = [s.strip() for s in role_row["skills"].split(",")]
        required_set = {s.lower() for s in required}

        have = [s for s in required if s.lower() in user_set]
        missing = [s for s in required if s.lower() not in user_set]

        readiness = round(len(have) / len(required) * 100, 1) if required else 0.0

        return {
            "role": role_name,
            "skills_have": have,
            "skills_missing": missing,
            "readiness_percent": readiness,
        }

    # ------------------------------------------------------------------
    # ADVANCED — Multi-algorithm benchmarking
    # ------------------------------------------------------------------
    def score_all_algorithms(self, user_skills, use_fuzzy: bool = True) -> pd.DataFrame:
        """
        Score every role against the user profile using three different
        similarity algorithms, for side-by-side benchmarking:
            - Cosine Similarity   (TF-IDF weighted, magnitude-invariant)
            - Jaccard Similarity  (simple set overlap, binary)
            - Euclidean Score     (1 / (1 + distance), magnitude-sensitive)
        """
        effective_skills = user_skills
        if use_fuzzy:
            effective_skills = [c[1] for c in self.correct_skills(user_skills)]

        user_vector_text = self._normalize(",".join(effective_skills))
        user_vector = self.vectorizer.transform([user_vector_text])

        cosine_scores = cosine_similarity(user_vector, self.role_matrix).flatten()
        eucl_dist = euclidean_distances(user_vector, self.role_matrix).flatten()
        euclidean_scores = 1 / (1 + eucl_dist)

        user_set = {s.strip().lower() for s in effective_skills}
        jaccard_scores = []
        for skills_list in self.df["skills_list"]:
            role_set = {s.strip().lower() for s in skills_list}
            union = user_set | role_set
            inter = user_set & role_set
            jaccard_scores.append(len(inter) / len(union) if union else 0.0)

        out = self.df[["role", "category"]].copy()
        out["cosine"] = np.round(cosine_scores, 4)
        out["jaccard"] = np.round(jaccard_scores, 4)
        out["euclidean_score"] = np.round(euclidean_scores, 4)
        return out.sort_values("cosine", ascending=False).reset_index(drop=True)

    def similarity_matrix(self) -> pd.DataFrame:
        """Role-vs-role cosine similarity matrix."""
        sim = cosine_similarity(self.role_matrix)
        return pd.DataFrame(sim, index=self.df["role"], columns=self.df["role"])

    def tfidf_top_terms(self, role_name: str, top_n: int = 8) -> pd.DataFrame:
        """Highest-weighted TF-IDF terms for a given role."""
        idx = self.df.index[self.df["role"] == role_name][0]
        row = self.role_matrix[idx].toarray().flatten()
        terms = np.array(self.vectorizer.get_feature_names_out())
        order = np.argsort(row)[::-1][:top_n]
        return pd.DataFrame({"skill": terms[order], "tfidf_weight": row[order]})


if __name__ == "__main__":
    engine = TechStackRecommender()
    sample = ["Pyhton", "Cloud Computing", "Automaton"]  # deliberate typos
    print(f"\nSample input skills (with typos): {sample}")
    print("\nFuzzy corrections:")
    for raw, fixed, score in engine.correct_skills(sample):
        print(f"  '{raw}' -> '{fixed}' (confidence: {score})")

    result = engine.recommend(sample, top_n=3)
    print("\nRecommendations:")
    print(result.to_string(index=False))

    top_role = result.iloc[0]["role"]
    print(f"\nExplainability for top role '{top_role}':")
    print(engine.explain(sample, top_role).to_string(index=False))

    print(f"\nSkill gap for '{top_role}':")
    gap = engine.skill_gap(sample, top_role)
    print(gap)
