## 🧭 AI Tech Stack Recommender — Advanced Edition

Content-Based Filtering · TF-IDF · Cosine Similarity · Fuzzy Matching · Explainable AI

DecodeLabs Industrial Training Program — Project 3 — Batch 2026

## Project Screenshots

https://github.com/user-attachments/assets/2f5ccd7a-bc72-4f1d-868a-4853e1bf99ed

https://github.com/user-attachments/assets/705c1042-02ee-4bb4-84b6-988e1220e807

https://github.com/user-attachments/assets/ec272742-8f01-4122-8e90-5be20807719b

https://github.com/user-attachments/assets/552cf534-5dec-4b8c-a033-17967f7400b6

<img width="1842" height="777" alt="image" src="https://github.com/user-attachments/assets/ad743c85-f705-43ed-9338-b460523aac1f" />


## 📌 Overview

An AI-powered recommendation engine that maps a user's raw skills to the most
relevant tech career paths — using Content-Based Filtering with
TF-IDF vectorization and Cosine Similarity, wrapped in a Streamlit
web app, a FastAPI REST service, and a CLI.

Unlike Collaborative Filtering (which needs historical user behavior data),
this engine works from day one with zero interaction history — matching
the intrinsic properties of a skillset directly against role requirements.

ProblemPersonalize tech career/stack recommendations from raw skill inputApproachContent-Based Filtering (Pattern Alignment)Core TechniqueTF-IDF Weighting + Cosine SimilarityDataset22 tech job roles across 7 categories, 6–8 core skills eachOutputRanked Top-N recommendations, explainability breakdown, skill-gap report


## 🆕 What's New in the Advanced Edition

FeatureWhy it matters🔤 Fuzzy skill correctionTypos and near-matches ("Pyhton", "ReactJS", "Web Design") auto-correct to the nearest known skill via rapidfuzz, solving the vocabulary-mismatch problem🔍 Explainability engineexplain() breaks down exactly which skills drove a role's match score, and by how much — not a black box🎯 Skill-gap analysisskill_gap() shows exactly what's missing to fully qualify for a role — turns the tool into a career roadmap, not just a ranking📊 Multi-algorithm benchmarkingCosine vs. Jaccard vs. Euclidean scoring compared side-by-side (algorithms_compare.py) with empirical proof of why Cosine wins🕸️ Skill–role network graphVisualizes the bipartite graph connecting skills to roles (networkx)🌐 FastAPI REST serviceProduction-style /recommend, /explain, /skill-gap endpoints with Swagger docs🐳 DockerizedOne image, runs either the Streamlit app or the API✅ Unit testspytest suite covering the core engine (tests/test_recommender.py)


## 🧠 Why Content-Based Filtering?


Collaborative Filtering — relies on community behavior ("users who picked X also picked Y"). Needs large historical interaction datasets. Suffers from cold start.
Content-Based Filtering — maps user preferences directly to item attributes. Works immediately for brand-new users and brand-new roles.


This project deliberately picks Content-Based Filtering to deliver value with zero historical data.


## ⚙️ How It Works — The IPO Pipeline

INPUT                     PROCESS                              OUTPUT
─────                     ───────                               ──────
Raw skills      ──►   1. Fuzzy-correct typos (rapidfuzz)   ──►  Ranked
(min. 3)               2. TF-IDF vectorize (shared vocab)       Top-N roles
                       3. Cosine similarity vs. 22 roles         + match %
                       4. Sort + truncate to Top-N               + explanation
                                                                  + skill gaps

Why TF-IDF? Rare, specific skills (Kubernetes) carry more weight than
generic ones (Git) — Term Frequency–Inverse Document Frequency handles this
automatically.

Why Cosine Similarity over Euclidean Distance? Euclidean distance is
sensitive to vector magnitude — a user with 8 skills would look "far" from
a role with only 4, even with perfect alignment. Cosine similarity measures
the angle between vectors, staying invariant to profile length — the
industry standard for text/skill-based similarity. See algorithms_compare.py
for the empirical comparison.

Cold Start handling:

ScenarioStrategyUser Cold Start (no skills)Trending Fallback — most popular roles in the datasetItem Cold Start (new role added)Not an issue — content-based scoring works instantly on new metadata


📁 Project Structure

iris_project3/
├── app/
│   └── app.py                       # Streamlit web app (3 tabs)
├── api.py                           # FastAPI REST service
├── recommender.py                   # Core engine: TF-IDF, cosine sim, fuzzy match, explainability
├── train_pipeline.py                # Full pipeline + visualization generator
├── algorithms_compare.py            # Cosine vs. Jaccard vs. Euclidean benchmark
├── predict_cli.py                   # Command-line predictor
├── tests/
│   └── test_recommender.py          # pytest unit tests
├── data/
│   ├── raw_skills.csv               # 22 roles → skills dataset
│   └── sample_recommendations.json
├── visuals/                          # 9 generated plots (PNG)
├── Dockerfile
├── requirements.txt
└── README.md


## 🚀 Getting Started

Prerequisites


Python 3.11 (or 3.9+)
pip


## 1. Install dependencies

bashpip install -r requirements.txt

## 2. Run the full pipeline (regenerates all visuals)

bashpython3 train_pipeline.py

## 3. CLI quick recommendation

bashpython3 predict_cli.py "Python" "Machine Learning" "SQL" "Statistics"

## Top 3 recommended career paths:

  1. Data Scientist            — 58.7% match
  2. Machine Learning Engineer — 46.4% match
  3. AI Research Scientist     — 42.1% match

## 4. Launch the Streamlit web app

bashstreamlit run app/app.py

Opens at http://localhost:8501 — three tabs: Live Recommendation, Model
Insights, Batch Recommendation (CSV upload).

## 5. Launch the REST API

bashuvicorn api:app --reload --port 8000

Interactive Swagger docs at http://localhost:8000/docs.

## 6. Run tests

bashpytest tests/

## 7. Run with Docker

bashdocker build -t tech-stack-recommender .
docker run -p 8501:8501 tech-stack-recommender          # Streamlit
docker run -p 8000:8000 tech-stack-recommender uvicorn api:app --host 0.0.0.0 --port 8000   # API


##🔌 API Reference (excerpt)

EndpointMethodDescription/GETHealth check/recommendPOSTGet Top-N ranked role recommendations for a skill list/explainPOSTBreakdown of which skills drove a specific role's score/skill-gapPOSTSkills still missing for a target role

Example request body for /recommend:

json{
  "skills": ["Python", "Cloud Computing", "Automation"],
  "top_n": 3,
  "use_fuzzy": true
}


## 📈 Visualizations (visuals/)


Roles-per-category distribution
Role-to-role cosine similarity heatmap (22×22)
TF-IDF top-weighted terms for a sample role
Match score comparison for a sample user profile
Cosine vs. Euclidean — why cosine wins at scale
Skill–role bipartite network graph
Explainability score breakdown
Fuzzy correction before/after demo
Cosine vs. Jaccard vs. Euclidean algorithm comparison



## 🛠️ Tech Stack

Python · scikit-learn (TF-IDF, Cosine Similarity) · rapidfuzz (fuzzy matching) ·
pandas · NumPy · Matplotlib · Seaborn · NetworkX · Streamlit ·
FastAPI + Uvicorn · pytest · Docker


## 🎯 Key Learnings


Building a recommendation engine on the IPO (Input → Process → Output) architecture
Vector-mapping raw text into a shared numerical vocabulary space
Why TF-IDF beats simple binary overlap for feature weighting
Why Cosine Similarity is the industry standard over Euclidean distance for text-based similarity
Designing cold-start-resilient recommendation logic
Adding explainability and skill-gap analysis on top of a similarity engine
Shipping the same core engine through three interfaces: web app, REST API, CLI
Containerizing an ML application for portable deployment



## 👩‍💻 Author

Soni 
## 🔗 LinkedIn: linkedin.com/in/soni-devi-131a9938b

## 💻 GitHub: github.com/Soni875612

### 🧩 LeetCode: leetcode.com/u/soni_2007

📄 License

Developed as part of the DecodeLabs Industrial Training Kit. Free to use for learning and reference purposes.
