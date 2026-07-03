# 🧭 AI Tech Stack Recommender — Advanced Content-Based Filtering Engine

An advanced **AI Recommendation System** that maps a user's raw skills to the most relevant tech career paths, using **TF-IDF vectorization**, **Cosine Similarity**, **fuzzy skill correction**, **explainable AI**, and **skill-gap analysis**. Built as **Project 3 (Advanced Edition)** of the **DecodeLabs Industrial Training Program (Batch 2026)**.

Unlike collaborative filtering (which needs historical user behavior), this engine uses **Content-Based Filtering** — it works from day one, with zero interaction history, by matching the *intrinsic properties* of a user's skillset directly against job role requirements.

---

## ✨ What Makes This "Advanced"

| Feature | Description |
|---|---|
| 🔤 **Fuzzy Skill Correction** | Typos and naming variants ("Pyhton", "Dokcer") are auto-corrected to the nearest known skill using `rapidfuzz`, solving the vocabulary-mismatch problem directly |
| 🔍 **Explainable AI** | Every recommendation can be broken down skill-by-skill to show exactly *why* a role was suggested and how much each skill contributed |
| 🧩 **Skill-Gap Analysis** | Turns the engine into a career roadmap tool — shows what you already have vs. what to learn next, with a readiness % |
| ⚖️ **Multi-Algorithm Benchmarking** | Cosine Similarity is compared head-to-head against Jaccard Similarity and Euclidean Distance, with empirical proof of why cosine wins |
| 🌐 **REST API** | A production-style FastAPI service exposes the engine via `/recommend`, `/explain`, `/skill-gap`, `/compare-algorithms` |
| 🧪 **Automated Tests** | 12 unit tests (`pytest`) covering cold-start, ranking, fuzzy correction, explainability, and skill-gap logic |
| 🐳 **Containerized** | Ships with a `Dockerfile` for one-command deployment of either the web app or the API |
| 🕸️ **Network Graph** | Visualizes the skill ↔ role bipartite graph to reveal which skills bridge multiple career paths |

---

## 📌 Project Overview

| | |
|---|---|
| **Problem** | Personalize career/tech-stack recommendations from raw, possibly-messy user skill input |
| **Approach** | Content-Based Filtering (Pattern Alignment) with fuzzy correction and explainability |
| **Core Technique** | TF-IDF Weighting + Cosine Similarity |
| **Dataset** | 22 tech job roles across 7 categories, each mapped to 6–8 core skills |
| **Output** | Ranked Top-N recommendations with match %, contribution breakdown, and skill gaps |

---

## 🧠 Why Content-Based Filtering?

Recommendation systems generally diverge into two methodologies:

- **Collaborative Filtering** — driven by community behavior ("users who picked this also picked that"). Requires large historical interaction datasets.
- **Content-Based Filtering** — driven by item attributes, mapping user preferences directly to intrinsic properties. Works immediately, even for brand-new users and items.

This project deliberately uses **Content-Based Filtering**, extended with fuzzy correction so it's also resilient to messy, real-world input.

---

## ⚙️ How It Works — The IPO Architecture

**1. Input (User State)**
- Captures the user's raw skills (minimum 3 required)
- Each skill is fuzzy-matched against the known vocabulary to correct typos/variants before scoring
- Item dataset (`raw_skills.csv`) maps 22 job roles to their core skill sets

**2. Process (Similarity Logic)**
- **Vector Mapping** — Skills are transformed into numerical vectors using `TfidfVectorizer`, sharing a single vocabulary space with the role dataset
- **TF-IDF Weighting** — rare, specific skills (e.g. `Kubernetes`) carry more weight than generic ones (e.g. `Git`)
- **Cosine Similarity** — measures the angle between vectors, making the score invariant to profile "length"

**3. Output (Ranked Recommendations)**
- **Sorting** — roles ranked by descending similarity score
- **Filtering** — truncated to Top-N to prevent choice overload
- **Explainability** — per-skill contribution breakdown for any recommended role
- **Skill Gap** — missing skills + readiness % for any target role

```
INPUT (Skills + Fuzzy Correction) → PROCESS (TF-IDF + Cosine Similarity) → OUTPUT (Top-N + Explainability + Skill Gap)
```

---

## ⚖️ Algorithm Benchmark — Why Cosine Wins

The engine can score every role using three algorithms side-by-side (`score_all_algorithms()`):

| Algorithm | Sensitive to profile length? | Uses TF-IDF weighting? |
|---|---|---|
| **Cosine Similarity** (used) | ❌ No — magnitude-invariant | ✅ Yes |
| Jaccard Similarity | ❌ No | ❌ No (binary overlap only) |
| Euclidean Distance | ✅ Yes — penalizes longer profiles | ❌ No |

Across 4 benchmark profiles, Cosine and Jaccard agreed on the #1 pick only 3/4 times — demonstrating that TF-IDF weighting genuinely changes outcomes, not just scores. See `visuals/09_algorithm_comparison.png`.

---

## ❄️ Handling the Cold Start Problem

| Scenario | Strategy |
|---|---|
| **User Cold Start** (no skills provided) | Trending Fallback — returns the most popular roles in the dataset |
| **Item Cold Start** (new role added) | Not an issue — content-based filtering recommends new items instantly, with no historical data required |
| **Vocabulary Mismatch** (typos, synonyms) | Fuzzy correction via `rapidfuzz` maps input to the nearest known skill before scoring |

---

## 🌐 Live Web Application

An interactive **Streamlit** app (`app/app.py`) with **five** tabs:

1. **🎯 Live Recommendation** — Free-text skill entry with live fuzzy-correction feedback and Top-N ranked matches
2. **🔍 Explainability & Skill Gap** — See exactly which skills drove a recommendation, and what to learn next to fully qualify for any role
3. **⚖️ Algorithm Comparison** — Cosine vs. Jaccard vs. Euclidean, side-by-side, for your own profile
4. **📊 Model Insights** — Dataset composition, role-to-role similarity heatmap, TF-IDF term weights
5. **📁 Batch Recommendation** — Upload a CSV of multiple users and download bulk recommendations

---

## 🔌 REST API

A FastAPI service (`api.py`) exposes the same engine programmatically:

| Endpoint | Method | Description |
|---|---|---|
| `/roles` | GET | List all job roles |
| `/skills` | GET | List the known skill vocabulary |
| `/recommend` | POST | Get Top-N recommendations (with fuzzy corrections shown) |
| `/explain` | POST | Get skill-by-skill contribution breakdown for a role |
| `/skill-gap` | POST | Get readiness % and missing skills for a role |
| `/compare-algorithms` | POST | Score all roles with Cosine, Jaccard, and Euclidean |

```bash
uvicorn api:app --reload --port 8000
# then visit http://localhost:8000/docs for interactive Swagger UI
```

---

## 📈 Visualizations

The `visuals/` folder contains:
1. Roles-per-category distribution
2. Role-to-role cosine similarity heatmap (22×22)
3. TF-IDF top-weighted terms for a sample role
4. Match score comparison for a sample user profile
5. Why cosine similarity outperforms Euclidean distance at scale
6. Skill ↔ Role bipartite network graph
7. Explainability breakdown for a sample recommendation
8. Fuzzy correction demo (typos → corrected skills)
9. Cosine vs. Jaccard vs. Euclidean algorithm comparison

---

## 📁 Project Structure

```
iris_project3/
├── app/
│   └── app.py                      # Streamlit web application (5 tabs)
├── data/
│   ├── raw_skills.csv              # Job roles → skills dataset
│   └── sample_recommendations.json # Output from sample pipeline run
├── tests/
│   └── test_recommender.py         # 12 pytest unit tests
├── visuals/                        # All generated plots (PNG)
├── recommender.py                  # Core engine: TF-IDF, fuzzy match, explainability, skill-gap
├── train_pipeline.py               # Full IPO pipeline + visualization generator
├── algorithms_compare.py           # Cosine vs Jaccard vs Euclidean benchmark
├── predict_cli.py                  # Command-line predictor (--explain, --gap flags)
├── api.py                          # FastAPI REST API
├── Dockerfile                      # Containerized deployment
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation & Usage

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the full pipeline (generates all 9 visuals)**
```bash
python3 train_pipeline.py
```

**3. Benchmark similarity algorithms**
```bash
python3 algorithms_compare.py
```

**4. Get a recommendation from the terminal**
```bash
python3 predict_cli.py "Pyhton" "SQL" "Machine Learning" --explain --gap "Data Scientist"
```

**5. Launch the interactive web app**
```bash
streamlit run app/app.py
```

**6. Launch the REST API**
```bash
uvicorn api:app --reload --port 8000
```

**7. Run the test suite**
```bash
pytest tests/ -v
```

**8. Run with Docker**
```bash
docker build -t tech-stack-recommender .
docker run -p 8501:8501 tech-stack-recommender          # Streamlit app
docker run -p 8000:8000 tech-stack-recommender uvicorn api:app --host 0.0.0.0 --port 8000  # API
```

---

## 🧪 Example — Full Advanced Output

```bash
$ python3 predict_cli.py "Pyhton" "SQL" "Machine Learning" --explain --gap "Data Scientist"

Fuzzy corrections applied:
  'Pyhton' -> 'Python' (83.3% confidence)

Top 3 recommended career paths:
  1. Machine Learning Engineer — 55.0% match
  2. AI Research Scientist     — 49.8% match
  3. Data Scientist            — 49.5% match

Explainability — why 'Machine Learning Engineer' ranked #1:
  [✓] Machine Learning     contribution: 0.8448
  [✓] Python               contribution: 0.178
  [✗] SQL                  contribution: 0.0

Skill gap analysis for 'Data Scientist':
  Readiness: 37.5%
  Have:    Python, SQL, Machine Learning
  Missing: Statistics, Data Analysis, Pandas, NumPy, Data Visualization
```

---

## 🛠️ Tech Stack

`Python` · `scikit-learn (TF-IDF, Cosine Similarity)` · `rapidfuzz` · `networkx` · `FastAPI` · `pytest` · `Docker` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Streamlit`

---

## 🎯 Key Learnings

- Building a recommendation engine on the IPO (Input–Process–Output) architecture
- Vector mapping raw text into a shared numerical vocabulary space
- Why TF-IDF outperforms binary overlap, and why Cosine Similarity beats Euclidean Distance at scale — proven empirically, not just asserted
- Solving real-world vocabulary mismatch with fuzzy string matching
- Making a "black box" ML system explainable and actionable (skill-gap roadmap)
- Shipping the same core engine through three interfaces: CLI, web app, and REST API
- Writing automated tests for an ML pipeline
- Containerizing an ML application for deployment

---

## 👩‍💻 Author

**Soni Devi**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/soni-devi-131a9938b/)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/Soni875612)
[![LeetCode](https://img.shields.io/badge/LeetCode-Profile-orange?logo=leetcode)](https://leetcode.com/u/soni_2007/)

- 🔗 **LinkedIn:** [linkedin.com/in/soni-devi-131a9938b](https://www.linkedin.com/in/soni-devi-131a9938b/)
- 💻 **GitHub:** [github.com/Soni875612](https://github.com/Soni875612)
- 🧩 **LeetCode:** [leetcode.com/u/soni_2007](https://leetcode.com/u/soni_2007/)

---

## 📄 License

Developed as part of the **DecodeLabs Industrial Training Kit**. Free to use for learning and reference purposes.
