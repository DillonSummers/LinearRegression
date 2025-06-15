"""Linear regression pipeline for the polymer dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

from features import shifted_log1p

def build_preprocessor(columns: Iterable[str]) -> ColumnTransformer:
    """Builds the preprocessing transformer for the model."""
    orig_log = ["max_fft_value", "mayo_lewis"]
    orig_cube = ["harwoods_blockiness"]
    orig_square = ["sum_fft_value"]
    orig_yj1 = ["mean_block_size", "std_block_size"]
    orig_yj2 = ["min_charge", "max_charge"]

    log_cols = [c for c in orig_log if c in columns]
    cube_cols = [c for c in orig_cube if c in columns]
    square_cols = [c for c in orig_square if c in columns]
    yeoj_cols_size = [c for c in orig_yj1 if c in columns]
    yeoj_cols_charge = [c for c in orig_yj2 if c in columns]

    other_cols = [c for c in columns if c not in (log_cols + cube_cols + square_cols + yeoj_cols_size + yeoj_cols_charge)]

    transformers = []
    if log_cols:
        transformers.append(("log", FunctionTransformer(shifted_log1p, feature_names_out="one-to-one"), log_cols))
    if cube_cols:
        transformers.append(("cube", FunctionTransformer(np.cbrt, feature_names_out="one-to-one"), cube_cols))
    if square_cols:
        transformers.append(("square", FunctionTransformer(np.square, feature_names_out="one-to-one"), square_cols))
    if yeoj_cols_size:
        transformers.append(("yj1", PowerTransformer(method="yeo-johnson", standardize=False), yeoj_cols_size))
    if yeoj_cols_charge:
        transformers.append(("yj2", PowerTransformer(method="yeo-johnson", standardize=False), yeoj_cols_charge))
    transformers.append(("pass", "passthrough", other_cols))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def train(df: pd.DataFrame, target: str, model_path: Path, plots_dir: Path) -> None:
    """Trains a linear regression model and saves artifacts."""
    X = df.iloc[:, 9:]
    y = df[target]

    initial_drop = ["max_length", "min_block_size"]
    X = X.drop(columns=initial_drop, errors="ignore")

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    target_corr = X.join(y).corr()[y.name].abs()
    multi_drop = {i if target_corr[i] < target_corr[j] else j for i in upper.index for j in upper.columns if upper.loc[i, j] > 0.90}
    X = X.drop(columns=list(multi_drop), errors="ignore")

    preprocessor = build_preprocessor(X.columns)

    model = Pipeline([
        ("prep", preprocessor),
        ("scale", StandardScaler()),
        ("lr", LinearRegression()),
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    print(f"Target: {target}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    plt.plot(lims, lims, linestyle="--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs. Predicted {target}")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{target}_scatter.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        X.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="black",
    )
    plt.title("Correlation Matrix of Features")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{target}_ff.png", dpi=300)
    plt.close()

    pearson_corr = pd.concat([X, y], axis=1).corr()

    plt.figure(figsize=(4.5, 6))
    sns.heatmap(
        pearson_corr.loc[X.columns, [y.name]],
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="black",
    )
    plt.tight_layout()
    plt.savefig(
        plots_dir / f"{target}_ft.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()

    model.fit(X, y)
    joblib.dump(model, model_path)


def train_all(df: pd.DataFrame, targets: Sequence[str], models_dir: Path, plots_dir: Path) -> None:
    """Trains models for multiple targets."""
    models_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        name = target.replace(" ", "_").lower()
        train(df, target, models_dir / f"lr_{name}.joblib", plots_dir / name)


def main() -> None:
    """Entry point for CLI usage."""
    df = pd.read_csv("data/combined_feature.csv")
    target_cols = [
        "Area AVG",
        "Area STD",
        "RG AVG",
        "RG STD",
        "RDF Peak",
        "Coordination at Minimum",
    ]
    train_all(df, target_cols, Path("models"), Path("plots"))


if __name__ == "__main__":
    main()
