"""
DecodeLabs - Project 3 (Advanced)
FastAPI REST API — production-style access to the recommendation engine.

Run:
    uvicorn api:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive Swagger UI.
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from recommender import TechStackRecommender, MIN_INPUT_SKILLS

app = FastAPI(
    title="DecodeLabs Tech Stack Recommender API",
    description="Content-based AI recommendation engine (TF-IDF + Cosine Similarity) "
                "that maps user skills to the most relevant tech career paths.",
    version="2.0.0",
)

engine = TechStackRecommender()


class RecommendRequest(BaseModel):
    skills: List[str] = Field(..., min_items=1, example=["Python", "Cloud Computing", "Automation"])
    top_n: int = Field(3, ge=1, le=10)
    use_fuzzy: bool = True


class ExplainRequest(BaseModel):
    skills: List[str]
    role: str
    use_fuzzy: bool = True


@app.get("/")
def root():
    return {
        "service": "DecodeLabs Tech Stack Recommender API",
        "version": "2.0.0",
        "endpoints": ["/roles", "/skills", "/recommend", "/explain", "/skill-gap", "/compare-algorithms"],
    }


@app.get("/roles")
def list_roles():
    """List all job roles in the dataset."""
    return engine.df[["role", "category", "skills"]].to_dict(orient="records")


@app.get("/skills")
def list_skills():
    """List the full known-skill vocabulary (used for fuzzy correction)."""
    return {"count": len(engine.known_skills), "skills": sorted(engine.known_skills)}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if len(req.skills) < MIN_INPUT_SKILLS:
        raise HTTPException(
            status_code=400,
            detail=f"Please provide at least {MIN_INPUT_SKILLS} skills for accurate matching.",
        )
    result = engine.recommend(req.skills, top_n=req.top_n, use_fuzzy=req.use_fuzzy)
    corrections = engine.correct_skills(req.skills) if req.use_fuzzy else []
    return {
        "input_skills": req.skills,
        "fuzzy_corrections": [
            {"original": o, "corrected": c, "confidence": s} for o, c, s in corrections
        ],
        "recommendations": result.to_dict(orient="records"),
    }


@app.post("/explain")
def explain(req: ExplainRequest):
    valid_roles = engine.df["role"].tolist()
    if req.role not in valid_roles:
        raise HTTPException(status_code=404, detail=f"Unknown role '{req.role}'.")
    breakdown = engine.explain(req.skills, req.role, use_fuzzy=req.use_fuzzy)
    return {"role": req.role, "breakdown": breakdown.to_dict(orient="records")}


@app.post("/skill-gap")
def skill_gap(req: ExplainRequest):
    valid_roles = engine.df["role"].tolist()
    if req.role not in valid_roles:
        raise HTTPException(status_code=404, detail=f"Unknown role '{req.role}'.")
    return engine.skill_gap(req.skills, req.role, use_fuzzy=req.use_fuzzy)


@app.post("/compare-algorithms")
def compare_algorithms(req: RecommendRequest):
    if len(req.skills) < MIN_INPUT_SKILLS:
        raise HTTPException(
            status_code=400,
            detail=f"Please provide at least {MIN_INPUT_SKILLS} skills for accurate matching.",
        )
    result = engine.score_all_algorithms(req.skills, use_fuzzy=req.use_fuzzy)
    return result.head(req.top_n).to_dict(orient="records")
