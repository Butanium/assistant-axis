"""
Prepare experiment data for the Quarto report.

Reads all judgment JSONL files and raw result files, structures them into
clean DataFrames, and exports to article/data/ as parquet.
"""

import json
import sys
from pathlib import Path

import pandas as pd

PIPELINE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = PIPELINE_DIR / "results"
JUDGMENTS_DIR = RESULTS_DIR / "judgments"
DATA_DIR = Path(__file__).parent.parent / "data"


def load_judgments() -> pd.DataFrame:
    """Load all judgment JSONL files into a DataFrame."""
    rows = []
    for path in sorted(JUDGMENTS_DIR.glob("*_judgments.jsonl")):
        # Classify source based on filename prefix
        source = "constitution" if path.stem.startswith("constitution_") else "6-prompt"
        with open(path) as f:
            for line in f:
                d = json.loads(line.strip())
                scores = d.get("scores", {})
                rows.append({
                    "model": d.get("model", "?").split("/")[-1],
                    "persona": d.get("persona", "?"),
                    "condition": d["condition"],
                    "coefficient": d["coefficient"],
                    "trait": d.get("trait", "unknown"),
                    "persona_strength": scores.get("persona_strength", -1),
                    "assistant_compliance": scores.get("assistant_compliance", -1),
                    "harmfulness": scores.get("harmfulness", -1),
                    "coherence": scores.get("coherence", -1),
                    "user_prompt": d.get("user_prompt", ""),
                    "source_file": path.stem,
                    "source": source,
                })
    return pd.DataFrame(rows)


def load_raw_samples() -> pd.DataFrame:
    """Load raw model outputs from result files for the sample explorer."""
    rows = []
    result_files = sorted(RESULTS_DIR.glob("persona_steering_*.json")) + sorted(RESULTS_DIR.glob("constitution_*.json"))
    for path in result_files:
        data = json.load(open(path))
        model = data["model"].split("/")[-1]
        persona = data["persona"]
        source = "constitution" if path.stem.startswith("constitution_") else "6-prompt"

        if "samples" in data:
            for s in data["samples"]:
                rows.append({
                    "model": model,
                    "persona": persona,
                    "condition": s["condition"],
                    "coefficient": s["coefficient"],
                    "trait": s.get("trait", "unknown"),
                    "user_prompt": s.get("user_prompt", ""),
                    "response": s["response"],
                    "source": source,
                })
        elif "results" in data:
            prompts_info = data.get("prompts", {})
            for condition, prompt_dict in data["results"].items():
                for label, coeff_dict in prompt_dict.items():
                    user_prompt = prompts_info.get(label, {}).get("prompt", label)
                    for coeff_str, response in coeff_dict.items():
                        coeff = float(coeff_str.split("=")[1])
                        rows.append({
                            "model": model,
                            "persona": persona,
                            "condition": condition,
                            "coefficient": coeff,
                            "trait": label,
                            "user_prompt": user_prompt,
                            "response": response,
                            "source": source,
                        })
    return pd.DataFrame(rows)


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Judgments
    judgments_df = load_judgments()
    judgments_df.to_parquet(DATA_DIR / "judgments.parquet", index=False)
    print(f"Judgments: {len(judgments_df)} rows → article/data/judgments.parquet")

    # Summary stats
    summary = judgments_df.groupby(
        ["model", "persona", "condition", "coefficient", "source"]
    ).agg(
        persona_strength_mean=("persona_strength", "mean"),
        persona_strength_std=("persona_strength", "std"),
        assistant_compliance_mean=("assistant_compliance", "mean"),
        assistant_compliance_std=("assistant_compliance", "std"),
        coherence_mean=("coherence", "mean"),
        coherence_std=("coherence", "std"),
        n=("persona_strength", "count"),
    ).reset_index()
    summary.to_parquet(DATA_DIR / "summary.parquet", index=False)
    summary.to_csv(DATA_DIR / "summary.csv", index=False)
    print(f"Summary: {len(summary)} rows → article/data/summary.parquet")

    # Raw samples
    samples_df = load_raw_samples()
    samples_df.to_parquet(DATA_DIR / "samples.parquet", index=False)
    print(f"Samples: {len(samples_df)} rows → article/data/samples.parquet")


if __name__ == "__main__":
    main()
