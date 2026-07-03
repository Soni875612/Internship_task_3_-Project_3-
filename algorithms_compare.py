"""
DecodeLabs - Project 3 (Advanced)
Algorithm Benchmark — Cosine vs. Jaccard vs. Euclidean

Demonstrates empirically why Cosine Similarity is the industry standard for
TF-IDF weighted, variable-length skill profiles, by comparing all three
algorithms across a battery of synthetic user profiles of varying length.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from recommender import TechStackRecommender

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
os.makedirs(VISUALS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


PROFILES = {
    "Short profile (3 skills)": ["Python", "SQL", "Machine Learning"],
    "Long profile, same interests (7 skills)": [
        "Python", "SQL", "Machine Learning", "Statistics",
        "Data Analysis", "Pandas", "NumPy",
    ],
    "Cloud-focused (3 skills)": ["AWS", "Docker", "Kubernetes"],
    "Security-focused (4 skills)": ["Networking", "Linux", "Security", "Scripting"],
}


def main():
    engine = TechStackRecommender()

    print("Benchmarking Cosine vs Jaccard vs Euclidean across sample profiles...\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    top1_agreement = 0
    total = 0

    for ax, (persona, skills) in zip(axes, PROFILES.items()):
        scores = engine.score_all_algorithms(skills).head(6)
        melted = scores.melt(
            id_vars=["role"], value_vars=["cosine", "jaccard", "euclidean_score"],
            var_name="algorithm", value_name="score"
        )
        sns.barplot(data=melted, x="score", y="role", hue="algorithm", ax=ax)
        ax.set_title(f"{persona}\nInput: {', '.join(skills)}", fontsize=9)
        ax.set_xlabel("Score")
        ax.set_ylabel("")
        ax.legend(fontsize=7)

        # Track whether cosine and jaccard agree on the #1 pick
        top_by_cosine = scores.sort_values("cosine", ascending=False).iloc[0]["role"]
        top_by_jaccard = scores.sort_values("jaccard", ascending=False).iloc[0]["role"]
        total += 1
        if top_by_cosine == top_by_jaccard:
            top1_agreement += 1

        print(f"{persona}:")
        print(scores[["role", "cosine", "jaccard", "euclidean_score"]].to_string(index=False))
        print()

    plt.tight_layout()
    out_path = os.path.join(VISUALS_DIR, "09_algorithm_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Top-1 agreement between Cosine and Jaccard across profiles: "
          f"{top1_agreement}/{total}")
    print(f"Chart saved to: {out_path}")

    print(
        "\nKey takeaway: Jaccard treats every skill as equally important and "
        "ignores TF-IDF weighting, so it can rank differently once profiles "
        "grow longer or contain generic vs. rare skills. Euclidean distance "
        "is magnitude-sensitive, so longer profiles are penalized even when "
        "the underlying *interests* are identical — this is exactly why "
        "Cosine Similarity is preferred for this use case."
    )


if __name__ == "__main__":
    main()
