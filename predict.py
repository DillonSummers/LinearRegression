"""CLI for predicting polymer properties using linear regression."""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib

from features import feature_dataframe_from_string

TARGET_MAP = {
    "area": "Area AVG",
    "rg": "RG AVG",
    "rdf": "RDF Peak",
    "coor": "Coordination at Minimum",
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
