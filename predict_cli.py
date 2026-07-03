"""
DecodeLabs - Project 3 (Advanced)
Quick CLI predictor — terminal-based Tech Stack Recommender with
fuzzy correction, explainability, and skill-gap analysis.

Usage:
    python3 predict_cli.py "Python" "Cloud Computing" "Automation"
    python3 predict_cli.py "Pyhton" "SQL" "Machine Learning" --top 5
    python3 predict_cli.py "Python" "SQL" "Machine Learning" --explain
    python3 predict_cli.py "Python" "SQL" "Machine Learning" --gap "Data Scientist"
    python3 predict_cli.py "Python" "SQL" "Machine Learning" --no-fuzzy
"""

import sys
import argparse
from recommender import TechStackRecommender, MIN_INPUT_SKILLS


def main():
    parser = argparse.ArgumentParser(
        description="DecodeLabs Tech Stack Recommender — CLI (Advanced)"
    )
    parser.add_argument("skills", nargs="+", help='Your skills, e.g. Python "Machine Learning" SQL')
    parser.add_argument("--top", type=int, default=3, help="Number of recommendations to return")
    parser.add_argument("--no-fuzzy", action="store_true", help="Disable fuzzy typo correction")
    parser.add_argument("--explain", action="store_true", help="Show why the #1 role was recommended")
    parser.add_argument("--gap", type=str, default=None, help="Show skill-gap analysis for a given role")
    args = parser.parse_args()

    if len(args.skills) < MIN_INPUT_SKILLS:
        print(f"Please provide at least {MIN_INPUT_SKILLS} skills for accurate matching.")
        print('Example: python3 predict_cli.py "Python" "Cloud Computing" "Automation"')
        sys.exit(1)

    use_fuzzy = not args.no_fuzzy
    engine = TechStackRecommender()

    if use_fuzzy:
        corrections = engine.correct_skills(args.skills)
        changed = [(o, c, s) for o, c, s in corrections if o.lower() != c.lower()]
        if changed:
            print("\nFuzzy corrections applied:")
            for o, c, s in changed:
                print(f"  '{o}' -> '{c}' ({s}% confidence)")

    result = engine.recommend(args.skills, top_n=args.top, use_fuzzy=use_fuzzy)

    print(f"\nInput skills: {', '.join(args.skills)}")
    print(f"\nTop {args.top} recommended career paths:\n")
    for i, row in result.iterrows():
        print(f"  {i + 1}. {row['role']}  —  {row['match_percent']}% match")
        print(f"     Category: {row['category']}")
        print(f"     Core skills: {row['skills']}\n")

    top_role = result.iloc[0]["role"]

    if args.explain:
        print(f"Explainability — why '{top_role}' ranked #1:\n")
        breakdown = engine.explain(args.skills, top_role, use_fuzzy=use_fuzzy)
        for _, row in breakdown.iterrows():
            marker = "✓" if row["present_in_role"] else "✗"
            print(f"  [{marker}] {row['skill']:20s} contribution: {row['contribution_weight']}")
        print()

    if args.gap:
        gap = engine.skill_gap(args.skills, args.gap, use_fuzzy=use_fuzzy)
        print(f"Skill gap analysis for '{args.gap}':")
        print(f"  Readiness: {gap['readiness_percent']}%")
        print(f"  Have:    {', '.join(gap['skills_have']) or '(none)'}")
        print(f"  Missing: {', '.join(gap['skills_missing']) or '(none — fully qualified!)'}\n")


if __name__ == "__main__":
    main()
