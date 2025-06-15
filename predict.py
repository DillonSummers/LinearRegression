"""CLI for predicting polymer properties using linear regression."""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from features import feature_dataframe_from_string, shifted_log1p

TARGET_MAP = {
    "area_avg": "Area AVG",
    "area_std": "Area STD",
    "rg_avg": "RG AVG",
    "rg_std": "RG STD",
    "rdf_peak": "RDF Peak",
    "coordination": "Coordination at Minimum",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict polymer properties")
    parser.add_argument("polymer", help="Polymer input list string")
    parser.add_argument("target", choices=TARGET_MAP.keys(), help="Target name")
    args = parser.parse_args()

    model_path = Path("models") / f"lr_{args.target}.joblib"
    model = joblib.load(model_path)

    df = feature_dataframe_from_string(args.polymer)
    X = df[model.feature_names_in_]
    pred = model.predict(X)[0]
    print(f"{args.target}: {pred:.4f}")


if __name__ == "__main__":
    main()
