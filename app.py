"""
DecodeLabs - Project 3 (Advanced): AI Recommendation Logic
Streamlit App — Tech Stack Recommender v2
"""

import os
import sys
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recommender import TechStackRecommender, MIN_INPUT_SKILLS  # noqa: E402

st.set_page_config(page_title="DecodeLabs | Tech Stack Recommender", page_icon="🧭", layout="wide")


@st.cache_resource
def load_engine():
    return TechStackRecommender()


engine = load_engine()
all_skills = sorted(engine.known_skills)

st.title("🧭 AI-Powered Tech Stack Recommender")
st.caption(
    "DecodeLabs · Project 3 (Advanced) · Content-Based Filtering with "
    "TF-IDF, Fuzzy Matching, Explainability & Skill-Gap Analysis"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Live Recommendation",
    "🔍 Explainability & Skill Gap",
    "⚖️ Algorithm Comparison",
    "📊 Model Insights",
    "📁 Batch Recommendation",
])

# ----------------------------------------------------------------------
# TAB 1 — Live Recommendation
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Find your ideal tech career path")
    st.write(
        f"Type or select at least **{MIN_INPUT_SKILLS} skills**. "
        "Typos are auto-corrected using fuzzy matching (e.g. 'Pyhton' → 'Python')."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        free_text = st.text_input(
            "Type skills, comma-separated (typos OK)",
            value="Pyhton, Cloud Computing, Automaton",
        )
        selected_skills = [s.strip() for s in free_text.split(",") if s.strip()]
        top_n = st.slider("Number of recommendations", 1, 8, 3)
        use_fuzzy = st.checkbox("Enable fuzzy typo correction", value=True)

    with col2:
        st.info(
            "💡 **How it works**\n\n"
            "Your skills are fuzzy-corrected, then converted into a TF-IDF "
            "weighted vector. Rare, specific skills carry more weight than "
            "generic ones. We compute **cosine similarity** — the angle "
            "between your vector and every role's vector — to rank matches."
        )

    if len(selected_skills) < MIN_INPUT_SKILLS:
        st.warning(f"Please provide at least {MIN_INPUT_SKILLS} skills to get accurate recommendations.")
    else:
        if use_fuzzy:
            corrections = engine.correct_skills(selected_skills)
            changed = [(o, c, s) for o, c, s in corrections if o.lower() != c.lower()]
            if changed:
                st.caption("✏️ Fuzzy corrections applied:")
                for o, c, s in changed:
                    st.caption(f"　'{o}' → **{c}** ({s}% confidence)")

        result = engine.recommend(selected_skills, top_n=top_n, use_fuzzy=use_fuzzy)

        st.success(f"Top {top_n} recommended career paths for your profile:")
        for i, row in result.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"### {i + 1}. {row['role']}")
                    st.caption(f"Category: {row['category']}")
                    st.write(f"**Core skills:** {row['skills']}")
                with c2:
                    st.metric("Match", f"{row['match_percent']}%")
                st.progress(min(int(row['match_percent']), 100))

        st.markdown("#### Match Score Comparison")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.barplot(data=result, x="match_percent", y="role", hue="role", ax=ax, legend=False, palette="viridis")
        ax.set_xlabel("Match %")
        ax.set_ylabel("")
        ax.set_xlim(0, 100)
        st.pyplot(fig)

# ----------------------------------------------------------------------
# TAB 2 — Explainability & Skill Gap
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Why was this recommended? What's missing?")

    exp_text = st.text_input(
        "Your skills (comma-separated)",
        value="Python, SQL, Machine Learning",
        key="exp_skills",
    )
    exp_skills = [s.strip() for s in exp_text.split(",") if s.strip()]

    if len(exp_skills) < MIN_INPUT_SKILLS:
        st.warning(f"Please provide at least {MIN_INPUT_SKILLS} skills.")
    else:
        role_choice = st.selectbox(
            "Explain a specific role (defaults to your #1 match)",
            options=["(Top match)"] + engine.df["role"].tolist(),
        )
        if role_choice == "(Top match)":
            role_choice = engine.recommend(exp_skills, top_n=1).iloc[0]["role"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**🔍 Skill Contribution — {role_choice}**")
            breakdown = engine.explain(exp_skills, role_choice)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.barplot(data=breakdown, x="contribution_weight", y="skill", hue="skill",
                        ax=ax, legend=False, palette="rocket")
            ax.set_xlabel("Contribution to Match Score")
            ax.set_ylabel("")
            st.pyplot(fig)
            st.dataframe(breakdown, use_container_width=True, hide_index=True)

        with col2:
            st.markdown(f"**🧩 Skill Gap — {role_choice}**")
            gap = engine.skill_gap(exp_skills, role_choice)
            st.metric("Readiness", f"{gap['readiness_percent']}%")
            st.progress(min(int(gap['readiness_percent']), 100))
            st.write("✅ **Skills you already have:**")
            st.write(", ".join(gap["skills_have"]) if gap["skills_have"] else "_None yet_")
            st.write("📚 **Skills to learn next:**")
            st.write(", ".join(gap["skills_missing"]) if gap["skills_missing"] else "_You're fully qualified!_")

# ----------------------------------------------------------------------
# TAB 3 — Algorithm Comparison
# ----------------------------------------------------------------------
with tab3:
    st.subheader("Cosine vs. Jaccard vs. Euclidean")
    st.write(
        "Compare how three different similarity algorithms rank the same "
        "user profile — this is why the engine uses **Cosine Similarity** "
        "as its primary scoring method."
    )

    algo_text = st.text_input(
        "Your skills (comma-separated)",
        value="Python, SQL, Machine Learning, Statistics",
        key="algo_skills",
    )
    algo_skills = [s.strip() for s in algo_text.split(",") if s.strip()]

    if len(algo_skills) < MIN_INPUT_SKILLS:
        st.warning(f"Please provide at least {MIN_INPUT_SKILLS} skills.")
    else:
        comp = engine.score_all_algorithms(algo_skills).head(6)
        melted = comp.melt(
            id_vars=["role"], value_vars=["cosine", "jaccard", "euclidean_score"],
            var_name="algorithm", value_name="score",
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(data=melted, x="score", y="role", hue="algorithm", ax=ax)
        ax.set_xlabel("Score")
        ax.set_ylabel("")
        st.pyplot(fig)
        st.dataframe(comp, use_container_width=True, hide_index=True)

        st.caption(
            "**Cosine Similarity** (TF-IDF weighted) rewards rare, specific skill overlap and "
            "is invariant to profile length. **Jaccard** treats every skill equally (binary "
            "overlap). **Euclidean** is sensitive to profile length — longer profiles can score "
            "lower even with identical interests."
        )

# ----------------------------------------------------------------------
# TAB 4 — Model Insights
# ----------------------------------------------------------------------
with tab4:
    st.subheader("Behind the Recommendation Engine")

    c1, c2, c3 = st.columns(3)
    c1.metric("Job Roles", len(engine.df))
    c2.metric("Categories", engine.df["category"].nunique())
    c3.metric("Unique Skills (vocabulary)", len(engine.vectorizer.get_feature_names_out()))

    st.markdown("**Roles per category**")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    cat_counts = engine.df["category"].value_counts()
    sns.barplot(x=cat_counts.values, y=cat_counts.index, hue=cat_counts.index, ax=ax1, legend=False, palette="crest")
    ax1.set_xlabel("Number of Roles")
    st.pyplot(fig1)

    st.markdown("**Role-to-role cosine similarity heatmap**")
    sim_df = engine.similarity_matrix()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(sim_df, cmap="mako", square=True, ax=ax2, cbar_kws={"label": "Cosine Similarity"})
    ax2.tick_params(axis="x", rotation=90, labelsize=6)
    ax2.tick_params(axis="y", labelsize=6)
    st.pyplot(fig2)

    st.markdown("**TF-IDF term weights for a selected role**")
    role_pick = st.selectbox("Choose a role", engine.df["role"].tolist(), index=0, key="insights_role")
    top_terms = engine.tfidf_top_terms(role_pick, top_n=8)
    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    sns.barplot(data=top_terms, x="tfidf_weight", y="skill", hue="skill", ax=ax3, legend=False, palette="flare")
    ax3.set_xlabel("TF-IDF Weight")
    ax3.set_ylabel("")
    st.pyplot(fig3)

# ----------------------------------------------------------------------
# TAB 5 — Batch Recommendation
# ----------------------------------------------------------------------
with tab5:
    st.subheader("Batch Recommendation")
    st.write(
        "Upload a CSV with `user_id` and `skills` columns to get Top-3 "
        "recommendations for every user at once, with fuzzy correction applied."
    )

    sample_csv = "user_id,skills\nu1,\"Pyhton, SQL, Machine Learning\"\nu2,\"JavaScript, React, CSS, UI Design\"\n"
    st.download_button("⬇️ Download sample CSV", sample_csv, file_name="sample_users.csv")

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            if not {"user_id", "skills"}.issubset(batch_df.columns):
                st.error("CSV must contain 'user_id' and 'skills' columns.")
            else:
                rows = []
                for _, r in batch_df.iterrows():
                    skills_list = [s.strip() for s in str(r["skills"]).split(",") if s.strip()]
                    if len(skills_list) < MIN_INPUT_SKILLS:
                        rows.append({"user_id": r["user_id"], "role": "N/A — needs more skills",
                                      "match_percent": None})
                        continue
                    rec = engine.recommend(skills_list, top_n=3)
                    for rank, row in rec.iterrows():
                        rows.append({
                            "user_id": r["user_id"],
                            "rank": rank + 1,
                            "role": row["role"],
                            "match_percent": row["match_percent"],
                        })
                out_df = pd.DataFrame(rows)
                st.success(f"Generated recommendations for {batch_df['user_id'].nunique()} users.")
                st.dataframe(out_df, use_container_width=True)

                csv_buffer = io.StringIO()
                out_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "⬇️ Download results as CSV",
                    csv_buffer.getvalue(),
                    file_name="batch_recommendations.csv",
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")

st.markdown("---")
st.caption("DecodeLabs Industrial Training Kit · Project 3 (Advanced) · AI Recommendation Logic")
