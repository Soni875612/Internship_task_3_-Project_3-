"""
DecodeLabs - Project 3: AI Recommendation Logic
================================================
Full IPO pipeline runner. Builds the recommendation engine, evaluates it
with sample user profiles, and generates all visualizations into /visuals.

Run:
    python3 train_pipeline.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from recommender import TechStackRecommender, MIN_INPUT_SKILLS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(VISUALS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    # ------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------
    section("INPUT — Loading dataset & building TF-IDF vector space")
    engine = TechStackRecommender()
    print(f"Loaded {len(engine.df)} job roles across "
          f"{engine.df['category'].nunique()} categories.")
    print(f"Vocabulary size (unique skills): {len(engine.vectorizer.get_feature_names_out())}")

    # ------------------------------------------------------------
    # PROCESS — Demo recommendations for a few sample user profiles
    # ------------------------------------------------------------
    section("PROCESS — Running sample user profiles through the engine")
    sample_profiles = {
        "Aspiring Data Scientist": ["Python", "Machine Learning", "Statistics", "SQL"],
        "Cloud-focused Engineer": ["Python", "Cloud Computing", "Automation"],
        "Frontend-leaning Developer": ["JavaScript", "React", "UI Design", "CSS"],
        "Security Enthusiast": ["Networking", "Linux", "Security", "Scripting"],
    }

    all_results = {}
    for persona, skills in sample_profiles.items():
        result = engine.recommend(skills, top_n=3)
        all_results[persona] = {
            "input_skills": skills,
            "recommendations": result.to_dict(orient="records"),
        }
        print(f"\n{persona} -> input: {skills}")
        print(result[["role", "match_percent"]].to_string(index=False))

    with open(os.path.join(DATA_DIR, "sample_recommendations.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ------------------------------------------------------------
    # OUTPUT — Visualizations
    # ------------------------------------------------------------
    section("OUTPUT — Generating visualizations")

    # 1. Role-skill distribution (skills per category)
    plt.figure(figsize=(9, 5))
    cat_counts = engine.df["category"].value_counts()
    sns.barplot(x=cat_counts.values, y=cat_counts.index, palette="crest")
    plt.title("Job Roles per Category in Dataset")
    plt.xlabel("Number of Roles")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "01_roles_per_category.png"), dpi=150)
    plt.close()

    # 2. Role-vs-role cosine similarity heatmap
    sim_df = engine.similarity_matrix()
    plt.figure(figsize=(14, 12))
    sns.heatmap(sim_df, cmap="mako", square=True, cbar_kws={"label": "Cosine Similarity"})
    plt.title("Role-to-Role Cosine Similarity Matrix")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "02_similarity_heatmap.png"), dpi=150)
    plt.close()

    # 3. TF-IDF top terms for a representative role
    plt.figure(figsize=(8, 5))
    top_terms = engine.tfidf_top_terms("Machine Learning Engineer", top_n=8)
    sns.barplot(data=top_terms, x="tfidf_weight", y="skill", palette="flare")
    plt.title("TF-IDF Weighted Skills — Machine Learning Engineer")
    plt.xlabel("TF-IDF Weight")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "03_tfidf_top_terms.png"), dpi=150)
    plt.close()

    # 4. Match score bar chart for one sample persona
    plt.figure(figsize=(8, 5))
    demo = engine.recommend(sample_profiles["Aspiring Data Scientist"], top_n=5)
    sns.barplot(data=demo, x="match_percent", y="role", palette="viridis")
    plt.title("Top-5 Recommendations — 'Aspiring Data Scientist' Profile")
    plt.xlabel("Match %")
    plt.ylabel("")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "04_sample_recommendation_scores.png"), dpi=150)
    plt.close()

    # 5. Cosine vs Euclidean illustration (why cosine wins at scale)
    plt.figure(figsize=(6, 6))
    origin = np.array([0, 0])
    a = np.array([2, 4])
    b = np.array([3, 6])  # same direction as a, different magnitude
    for vec, label, color in [(a, "User A (short profile)", "#2563eb"),
                               (b, "User B (long profile, same interests)", "#dc2626")]:
        plt.quiver(*origin, *vec, angles="xy", scale_units="xy", scale=1, color=color, width=0.008)
        plt.text(vec[0] * 1.05, vec[1] * 1.05, label, fontsize=8, color=color)
    plt.xlim(-1, 7)
    plt.ylim(-1, 8)
    plt.title("Why Cosine Similarity > Euclidean Distance\n(Same direction = same interests, regardless of profile length)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "05_cosine_vs_euclidean.png"), dpi=150)
    plt.close()

    # 6. Skill <-> Role bipartite network graph
    try:
        import networkx as nx
        G = nx.Graph()
        for _, row in engine.df.iterrows():
            G.add_node(row["role"], bipartite=0, kind="role")
            for skill in row["skills_list"]:
                G.add_node(skill, bipartite=1, kind="skill")
                G.add_edge(row["role"], skill)

        # Keep the graph readable: only show skills shared by 2+ roles
        skill_degrees = {n: d for n, d in G.degree() if G.nodes[n]["kind"] == "skill"}
        shared_skills = {n for n, d in skill_degrees.items() if d >= 2}
        role_nodes = {n for n, attr in G.nodes(data=True) if attr["kind"] == "role"}
        keep_nodes = role_nodes | shared_skills
        H = G.subgraph(keep_nodes)

        plt.figure(figsize=(15, 12))
        pos = nx.spring_layout(H, k=0.5, seed=42, iterations=60)
        role_list = [n for n in H.nodes if H.nodes[n]["kind"] == "role"]
        skill_list = [n for n in H.nodes if H.nodes[n]["kind"] == "skill"]
        nx.draw_networkx_nodes(H, pos, nodelist=role_list, node_color="#dc2626",
                                node_size=500, label="Job Role")
        nx.draw_networkx_nodes(H, pos, nodelist=skill_list, node_color="#2563eb",
                                node_size=200, label="Shared Skill")
        nx.draw_networkx_edges(H, pos, alpha=0.3)
        nx.draw_networkx_labels(H, pos, font_size=6)
        plt.title("Skill <-> Role Network\n(Skills shared by 2+ roles only, for readability)")
        plt.legend(scatterpoints=1)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, "06_skill_role_network.png"), dpi=150)
        plt.close()
    except ImportError:
        print("networkx not installed — skipping network graph visualization.")

    # 7. Explainability chart — skill contribution breakdown for a sample profile
    sample_skills = sample_profiles["Aspiring Data Scientist"]
    top_role_for_sample = engine.recommend(sample_skills, top_n=1).iloc[0]["role"]
    breakdown = engine.explain(sample_skills, top_role_for_sample)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=breakdown, x="contribution_weight", y="skill", palette="rocket")
    plt.title(f"Explainability — Why '{top_role_for_sample}' Was Recommended\n"
              f"(input: {', '.join(sample_skills)})")
    plt.xlabel("Contribution to Match Score")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "07_explainability_breakdown.png"), dpi=150)
    plt.close()

    # 8. Fuzzy correction demo (typos -> corrected skills)
    typo_input = ["Pyhton", "Cloud Computing", "Automaton", "Dokcer"]
    corrections = engine.correct_skills(typo_input)
    corr_df = pd.DataFrame(corrections, columns=["typed_skill", "corrected_skill", "confidence"])
    plt.figure(figsize=(7, 3.5))
    sns.barplot(data=corr_df, x="confidence", y="typed_skill", palette="crest")
    for i, row in corr_df.iterrows():
        plt.text(row["confidence"] + 1, i, f"-> {row['corrected_skill']}", va="center", fontsize=8)
    plt.title("Fuzzy Skill Correction — Handling Typos & Naming Variants")
    plt.xlabel("Match Confidence (%)")
    plt.ylabel("As Typed")
    plt.xlim(0, 115)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "08_fuzzy_correction_demo.png"), dpi=150)
    plt.close()

    print(f"\nAll visuals saved to: {VISUALS_DIR}")
    section("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
