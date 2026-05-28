from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_jsonl


METADATA_COLUMNS = {
    "id",
    "source",
    "target",
    "kd_useful",
    "ce_step_val_loss",
    "kd_step_val_loss",
    "delta_val_loss",
    # Label/debug scores are allowed in the probe file, but must not be used as
    # gate inputs. For grad_align labeling this value directly defines kd_useful.
    "grad_align_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CHAD usefulness gate.")
    parser.add_argument("--probe-file", required=True)
    parser.add_argument("--output-file", default="runs/gate/chad_gate.joblib")
    parser.add_argument("--metrics-file", default="runs/gate/metrics.json")
    parser.add_argument(
        "--model-type",
        choices=["logistic", "mlp", "ridge", "mlp_regressor", "gbm"],
        default="gbm",
        help=(
            "logistic/mlp: binary classifier on kd_useful. "
            "ridge/mlp_regressor/gbm: regression on continuous grad_align_score."
        ),
    )
    parser.add_argument("--feature-columns", nargs="*", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def is_number(value: object) -> bool:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
        return False
    return not math.isnan(float(value))


def infer_feature_columns(rows: list[dict[str, object]]) -> list[str]:
    columns = sorted({key for row in rows for key in row.keys()})
    features = []
    for column in columns:
        if column in METADATA_COLUMNS:
            continue
        if any(is_number(row.get(column)) for row in rows):
            features.append(column)
    return features


def build_matrix(rows: list[dict[str, object]], columns: list[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, math.nan)
            if value is None:
                value = math.nan
            values.append(float(value))
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float32)


REGRESSION_TYPES = {"ridge", "mlp_regressor", "gbm"}


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.probe_file)
    if not rows:
        raise ValueError("Probe file is empty.")

    feature_columns = args.feature_columns or infer_feature_columns(rows)
    if not feature_columns:
        raise ValueError("No numeric feature columns found.")

    x = build_matrix(rows, feature_columns)
    is_regression = args.model_type in REGRESSION_TYPES

    if is_regression:
        # Use continuous grad_align_score as target
        y = np.asarray(
            [float(row.get("grad_align_score", row.get("kd_useful", 0))) for row in rows],
            dtype=np.float32,
        )
    else:
        y = np.asarray([int(row["kd_useful"]) for row in rows], dtype=np.int64)
        if len(set(y.tolist())) < 2:
            raise ValueError("Gate training needs both helpful and harmful labels.")

    stratify = None if is_regression else (y if min(np.bincount(y)) >= 2 else None)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.seed, stratify=stratify,
    )

    if args.model_type == "logistic":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
    elif args.model_type == "mlp":
        estimator = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=1000, random_state=args.seed)
    elif args.model_type == "ridge":
        estimator = Ridge(alpha=1.0)
    elif args.model_type == "mlp_regressor":
        estimator = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu", max_iter=1000, random_state=args.seed)
    else:  # gbm
        estimator = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            min_samples_leaf=10, random_state=args.seed,
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("estimator", estimator),
    ])
    model.fit(x_train, y_train)

    gate_type = "regressor" if is_regression else "classifier"
    metrics: dict = {
        "probe_file": args.probe_file,
        "model_type": args.model_type,
        "gate_type": gate_type,
        "feature_columns": feature_columns,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
    }

    if is_regression:
        pred = model.predict(x_test)
        metrics["mae"] = float(mean_absolute_error(y_test, pred))
        metrics["r2"] = float(r2_score(y_test, pred))
        # Also report classification AUC by thresholding at 0
        y_binary = (y_test > 0).astype(int)
        if len(set(y_binary.tolist())) == 2:
            metrics["roc_auc_vs_zero"] = float(roc_auc_score(y_binary, pred))
        metrics["positive_ratio"] = float((y > 0).mean())
    else:
        pred = model.predict(x_test)
        pred_proba = model.predict_proba(x_test)[:, 1]
        metrics["accuracy"] = float(accuracy_score(y_test, pred))
        metrics["f1"] = float(f1_score(y_test, pred, zero_division=0))
        metrics["roc_auc"] = float(roc_auc_score(y_test, pred_proba)) if len(set(y_test.tolist())) == 2 else None
        metrics["positive_ratio"] = float(y.mean())

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": feature_columns, "gate_type": gate_type}, args.output_file)
    Path(args.metrics_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_file).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
