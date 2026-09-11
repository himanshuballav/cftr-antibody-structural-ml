#!/usr/bin/env python3
"""
10_run_signed_and_absolute_classifier.py

FINAL SUPERVISED MACHINE-LEARNING WORKFLOW

Purpose
-------
Run the same leakage-safe classifier independently on:

    1) Signed structural deltas:
           Delta_feature = Human_feature - Rhesus_feature

    2) Absolute structural deltas:
           AbsDelta_feature = abs(Human_feature - Rhesus_feature)

Input
-----
Workbook created by 09_build_ml_ready_dataset.py

Required sheets:
    ML_Ready_Signed
    ML_Ready_Absolute

Target:
    Cross_Reactive_Target
        0 = Non-cross-reactive / reference-specific
        1 = Cross-reactive

Predictors:
    Signed model   -> Delta_* only
    Absolute model -> AbsDelta_* only

Experimental ELISA measurements, AUC values, AUC ratios, thresholds,
and other phenotype-derived columns are NEVER included in X.

Why nested validation is used
-----------------------------
The dataset is small, so every observation is valuable. Outer Leave-One-Out
Cross-Validation (LOOCV) provides one held-out prediction for every clone.

Within EACH outer training set, the workflow performs:

    Step 1. Remove constant predictors.
    Step 2. Perform Spearman redundancy filtering using TRAINING X only.
    Step 3. Standardize predictors using TRAINING data only.
    Step 4. Fit L1-regularized logistic regression.
    Step 5. Use inner stratified CV to select regularization strength C.
    Step 6. Predict the completely held-out outer clone.

Putting Spearman filtering and scaling inside the sklearn Pipeline prevents
information from the held-out clone from entering model fitting.

Important distinction
---------------------
Predictive performance:
    Report ONLY from the outer-LOOCV held-out predictions.

Descriptive interpretation:
    Full-dataset correlations, top-feature boxplots, Mann-Whitney tests,
    final full-data coefficients, and the saved final model are provided
    for interpretation and future application. They are NOT estimates of
    generalization performance.

Major outputs for each model
----------------------------
    Performance_Metrics.csv
    Performance_with_Bootstrap_CI.csv
    Outer_LOOCV_Predictions.csv
    Outer_LOOCV_Fold_Details.csv
    Feature_Stability.csv
    Outer_Model_Coefficients.csv
    Descriptive_Spearman_Correlation.csv
    Highly_Correlated_Feature_Pairs.csv
    Top10_Feature_Statistics.csv
    Top10_Stability_and_Effect_Size.csv
    Final_Model_Features.csv
    Final_Model.joblib
    Final_Model_Metadata.json
    Held_Out_LOOCV_Predictions.png / .pdf
    ROC_Curve.png / .pdf
    Confusion_Matrix.png / .pdf
    Feature_Stability.png / .pdf
    Precision_Recall_Curve.png / .pdf
    Feature_Selection_Across_Outer_Folds.png / .pdf
    Feature_Selection_Across_Outer_Folds.csv
    Coefficient_Sign_Consistency.png / .pdf
    Coefficient_Sign_Consistency.csv
    Clone_Level_Prediction_Diagnostics.csv
    Feature_Stability_vs_Coefficient_Direction.png / .pdf
    Full_Data_Spearman_Heatmap.png / .pdf
    Top10_Features_Combined.png / .pdf
    Top10_Stability_vs_Effect_Size.png / .pdf
    Analysis_Summary.txt

A top-level Model_Comparison.csv and Model_Comparison.xlsx are also created.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# 1. GLOBAL SETTINGS
# =============================================================================
# Central constants are kept here so the workflow is easy to audit and modify.
# =============================================================================

ID_COL = "Clone_ID"
TARGET_COL = "Cross_Reactive_Target"

SPEARMAN_THRESHOLD = 0.80
INNER_N_SPLITS = 4

C_GRID = [
    0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70,
    1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
]

RANDOM_SEED = 42
BOOTSTRAP_N = 5000
TOP_N = 10

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
    }
)


# =============================================================================
# 2. TRAINING-ONLY SPEARMAN REDUNDANCY FILTER
# =============================================================================
# This transformer is deliberately placed inside the sklearn Pipeline.
# Therefore, correlations are learned separately from each training fold and
# the held-out observation never contributes to feature filtering.
# =============================================================================

class SpearmanRedundancyFilter(BaseEstimator, TransformerMixin):
    """
    Training-only redundancy filter.

    A predictor is retained only if it is not strongly correlated
    (absolute Spearman rho >= threshold) with an already retained predictor.

    IMPORTANT:
    This transformer lives INSIDE the sklearn Pipeline, so fit() is called
    separately within each training fold.
    """

    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        X_num = X.apply(pd.to_numeric, errors="coerce")

        if X_num.isna().any().any():
            bad = X_num.columns[X_num.isna().any()].tolist()
            raise ValueError(
                "Missing/non-numeric values found during Spearman filtering: "
                + ", ".join(bad)
            )

        variances = X_num.var(axis=0, ddof=0)

        variable_features = variances[variances > 0].index.tolist()
        self.dropped_constant_features_ = variances[
            variances <= 0
        ].index.tolist()

        if not variable_features:
            raise ValueError("All predictors are constant in this training fold.")

        X_var = X_num[variable_features]

        corr = X_var.corr(method="spearman").abs().fillna(0.0)
        corr_array = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_array, 0.0)

        corr = pd.DataFrame(
            corr_array,
            index=corr.index,
            columns=corr.columns,
        )

        self.correlation_matrix_ = corr.copy()

        redundancy_score = corr.mean(axis=1)

        original_order = {
            feature: i for i, feature in enumerate(variable_features)
        }

        ordered = sorted(
            variable_features,
            key=lambda f: (redundancy_score.loc[f], original_order[f]),
        )

        kept = []

        for feature in ordered:
            if not kept:
                kept.append(feature)
                continue

            if corr.loc[feature, kept].max() < self.threshold:
                kept.append(feature)

        kept_set = set(kept)

        self.selected_features_ = [
            f for f in variable_features if f in kept_set
        ]

        self.dropped_correlated_features_ = [
            f for f in variable_features if f not in kept_set
        ]

        if not self.selected_features_:
            raise ValueError("Spearman filter retained zero predictors.")

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        return X.loc[:, self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


# =============================================================================
# 3. STATISTICAL HELPER FUNCTIONS
# =============================================================================

def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction without an extra dependency."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


def rank_biserial_from_u(u_stat: float, n0: int, n1: int) -> float:
    """
    Rank-biserial effect size oriented as CR minus Non-CR.

    scipy mannwhitneyu(x=CR, y=NonCR) returns U for CR.
    +1 means values tend to be higher in CR.
    -1 means values tend to be lower in CR.
    """
    return (2.0 * u_stat / (n1 * n0)) - 1.0


def bootstrap_metric_ci(
    y_true,
    y_prob,
    y_pred,
    metric_name,
    n_boot=BOOTSTRAP_N,
    seed=RANDOM_SEED,
):
    """
    Bootstrap the already-held-out outer-LOOCV predictions.

    This does NOT refit the model; it quantifies uncertainty in the set of
    held-out predictions.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)

        yt = y_true[idx]
        yp = y_prob[idx]
        yc = y_pred[idx]

        if len(np.unique(yt)) < 2:
            continue

        if metric_name == "ROC_AUC":
            value = roc_auc_score(yt, yp)
        elif metric_name == "Balanced_Accuracy":
            value = balanced_accuracy_score(yt, yc)
        elif metric_name == "MCC":
            value = matthews_corrcoef(yt, yc)
        else:
            raise ValueError(f"Unknown bootstrap metric: {metric_name}")

        if np.isfinite(value):
            values.append(float(value))

    if len(values) < 100:
        return np.nan, np.nan

    return tuple(np.percentile(values, [2.5, 97.5]))


# =============================================================================
# 4. MODEL PIPELINE
# =============================================================================
# Order matters:
#   Spearman filtering -> scaling -> L1 logistic regression
# All three operations are fit on training data only inside cross-validation.
# =============================================================================

def build_pipeline() -> Pipeline:
    """Create a fresh leakage-safe ML pipeline."""
    return Pipeline(
        steps=[
            (
                "spearman",
                SpearmanRedundancyFilter(threshold=SPEARMAN_THRESHOLD),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


# =============================================================================
# 5. INPUT VALIDATION AND LEAKAGE PROTECTION
# =============================================================================
# Only the expected structural prefixes are admitted into X.
# =============================================================================

def validate_input(
    df: pd.DataFrame,
    feature_prefix: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Validate one ML-ready sheet and return X, y, clone IDs, feature names."""
    for required in [ID_COL, TARGET_COL]:
        if required not in df.columns:
            raise ValueError(f"Required column '{required}' is missing.")

    if df[ID_COL].duplicated().any():
        duplicates = df.loc[df[ID_COL].duplicated(keep=False), ID_COL].tolist()
        raise ValueError(f"Duplicate Clone_ID values found: {duplicates}")

    y = pd.to_numeric(df[TARGET_COL], errors="raise").astype(int)

    unique_y = sorted(y.unique().tolist())
    if unique_y != [0, 1]:
        raise ValueError(
            f"{TARGET_COL} must contain both classes 0 and 1. Found: {unique_y}"
        )

    feature_columns = [
        str(c) for c in df.columns if str(c).startswith(feature_prefix)
    ]

    if not feature_columns:
        raise ValueError(
            f"No structural predictors beginning with '{feature_prefix}' found."
        )

    # Hard leakage protection.
    leakage = [
        c for c in feature_columns
        if "ELISA" in c.upper() or "EXPERIMENTAL_AUC" in c.upper()
    ]

    if leakage:
        raise ValueError(
            "Potential ELISA-derived leakage predictors entered X: "
            + ", ".join(leakage)
        )

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        missing = X.isna().sum()
        missing = missing[missing > 0]
        raise ValueError(
            "Missing/non-numeric predictor values detected:\n"
            + missing.to_string()
        )

    clone_ids = df[ID_COL].astype(str).str.strip()

    return X, y, clone_ids, feature_columns


# =============================================================================
# 6. PLOTTING / DESCRIPTIVE OUTPUT HELPERS
# =============================================================================

def save_heldout_prediction_plot(
    predictions: pd.DataFrame,
    outdir: Path,
    analysis_name: str,
) -> None:
    """
    Plot the probability assigned to the cross-reactive class for every
    completely held-out outer-LOOCV clone.

    Each point is generated from the model fitted WITHOUT that clone.

    The clones are ordered from lowest to highest predicted probability so the
    separation between predicted Non-CR and CR samples can be inspected easily.

    The dashed horizontal line at 0.50 is the classification threshold:
        probability < 0.50  -> predicted Non-CR
        probability >= 0.50 -> predicted CR

    Point labels are clone identifiers. Point color indicates the observed
    experimental class so correct and incorrect predictions can be inspected
    directly without changing the underlying probability values.
    """

    plot_df = (
        predictions
        .copy()
        .sort_values("Predicted_Probability", ascending=True)
        .reset_index(drop=True)
    )

    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot observed Non-CR and CR clones separately so the legend shows the
    # experimental class while the y-axis remains the held-out model probability.
    for observed_class, label in [
        (0, "Observed Non-CR"),
        (1, "Observed CR"),
    ]:
        mask = plot_df["Observed_Cross_Reactive"] == observed_class

        ax.scatter(
            x[mask],
            plot_df.loc[mask, "Predicted_Probability"],
            s=85,
            label=label,
            zorder=3,
        )

    # The classifier converts probability to class using 0.50.
    ax.axhline(
        0.50,
        linestyle="--",
        linewidth=1.5,
        label="Classification threshold = 0.50",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_df["Clone_ID"].astype(str),
        rotation=45,
        ha="right",
    )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Antibody Clone")
    ax.set_ylabel("Held-Out Predicted Probability of Cross-Reactivity")
    ax.set_title(
        f"{analysis_name}: Leakage-Safe Nested LOOCV Held-Out Predictions"
    )

    ax.legend(loc="best")
    fig.tight_layout()

    fig.savefig(
        outdir / "Held_Out_LOOCV_Predictions.png",
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        outdir / "Held_Out_LOOCV_Predictions.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def save_roc(y_true, y_prob, auc, outdir: Path, analysis_name: str) -> None:
    """Save ROC curve as PNG and PDF."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{analysis_name}: Outer-LOOCV ROC")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(outdir / "ROC_Curve.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir / "ROC_Curve.pdf", bbox_inches="tight")
    plt.close(fig)


def save_confusion(cm, outdir: Path, analysis_name: str) -> None:
    """Save simple confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-CR", "CR"])
    ax.set_yticklabels(["Non-CR", "CR"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(f"{analysis_name}: Confusion Matrix")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    fig.savefig(outdir / "Confusion_Matrix.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir / "Confusion_Matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def save_feature_stability_plot(
    stability: pd.DataFrame,
    outdir: Path,
    analysis_name: str,
) -> None:
    """Plot top features by L1 selection frequency."""
    plot_df = (
        stability
        .sort_values(
            [
                "L1_Selection_Frequency",
                "Mean_Absolute_Coefficient_Across_Outer_Models",
            ],
            ascending=False,
        )
        .head(15)
        .sort_values("L1_Selection_Frequency", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        plot_df["Feature"],
        plot_df["L1_Selection_Frequency"],
    )

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("L1 selection frequency across outer LOOCV models")
    ax.set_title(f"{analysis_name}: Feature Stability")

    fig.tight_layout()
    fig.savefig(outdir / "Feature_Stability.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir / "Feature_Stability.pdf", bbox_inches="tight")
    plt.close(fig)



def save_precision_recall_curve(
    y_true,
    y_prob,
    outdir: Path,
    analysis_name: str,
) -> float:
    """
    Save a precision-recall curve using the outer-LOOCV held-out probabilities.

    This is especially useful when class sizes are imbalanced because it
    focuses on the trade-off between precision and recall for the positive
    (cross-reactive) class.

    Returns:
        Average precision (AP).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        recall,
        precision,
        linewidth=2,
        label=f"Average precision = {ap:.3f}",
    )

    positive_fraction = float(np.mean(y_true))
    ax.axhline(
        positive_fraction,
        linestyle="--",
        linewidth=1,
        label=f"Positive-class prevalence = {positive_fraction:.3f}",
    )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{analysis_name}: Outer-LOOCV Precision-Recall Curve")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(
        outdir / "Precision_Recall_Curve.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "Precision_Recall_Curve.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    return float(ap)


def save_feature_selection_heatmap(
    coef_matrix: pd.DataFrame,
    folds: pd.DataFrame,
    stability: pd.DataFrame,
    outdir: Path,
    analysis_name: str,
    top_n: int = 20,
) -> None:
    """
    Visualize which structural features received non-zero L1 coefficients
    in each outer-LOOCV model.

    Rows:
        top stable features

    Columns:
        held-out clone for each outer fold

    Cell:
        1 = feature selected by L1 in that outer model
        0 = not selected

    This is useful for seeing whether feature importance is robust across
    held-out antibodies or depends strongly on specific clones.
    """
    ranked_features = (
        stability
        .sort_values(
            [
                "L1_Selection_Frequency",
                "Spearman_Retention_Frequency",
                "Mean_Absolute_Coefficient_Across_Outer_Models",
            ],
            ascending=False,
        )
        .head(min(top_n, len(stability)))
        ["Feature"]
        .tolist()
    )

    selected = (coef_matrix[ranked_features].abs() > 1e-12).astype(int).T

    held_out_labels = folds["Held_Out_Clone"].astype(str).tolist()
    selected.columns = held_out_labels

    selected.to_csv(
        outdir / "Feature_Selection_Across_Outer_Folds.csv",
        index=True,
        index_label="Feature",
    )

    fig, ax = plt.subplots(
        figsize=(max(12, 0.65 * len(held_out_labels) + 4),
                 max(8, 0.45 * len(ranked_features) + 3))
    )

    image = ax.imshow(
        selected.values,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(np.arange(len(held_out_labels)))
    ax.set_xticklabels(
        held_out_labels,
        rotation=45,
        ha="right",
        fontsize=9,
    )

    ax.set_yticks(np.arange(len(ranked_features)))
    ax.set_yticklabels(ranked_features, fontsize=9)

    ax.set_xlabel("Held-Out Clone")
    ax.set_ylabel("Structural Feature")
    ax.set_title(
        f"{analysis_name}: L1 Feature Selection Across Outer LOOCV Folds"
    )

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.03)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Not selected", "Selected"])

    fig.tight_layout()

    fig.savefig(
        outdir / "Feature_Selection_Across_Outer_Folds.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "Feature_Selection_Across_Outer_Folds.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def save_coefficient_sign_consistency(
    coef_matrix: pd.DataFrame,
    stability: pd.DataFrame,
    outdir: Path,
    analysis_name: str,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """
    Quantify coefficient-direction consistency across outer-LOOCV models.

    For each feature:
        - number/fraction of outer models with positive coefficient
        - number/fraction with negative coefficient
        - number/fraction with zero coefficient
        - dominant non-zero sign consistency

    A feature that is selected often AND keeps the same sign across outer
    models is easier to interpret mechanistically than a feature whose sign
    frequently flips.
    """
    records = []

    for feature in coef_matrix.columns:
        values = coef_matrix[feature].astype(float).to_numpy()

        positive = int(np.sum(values > 1e-12))
        negative = int(np.sum(values < -1e-12))
        zero = int(len(values) - positive - negative)
        nonzero = positive + negative

        if nonzero > 0:
            dominant_nonzero_fraction = max(positive, negative) / nonzero
            dominant_sign = "Positive" if positive >= negative else "Negative"
        else:
            dominant_nonzero_fraction = np.nan
            dominant_sign = "Never selected"

        records.append(
            {
                "Feature": feature,
                "Positive_Coefficient_Folds": positive,
                "Negative_Coefficient_Folds": negative,
                "Zero_Coefficient_Folds": zero,
                "Positive_Frequency_All_Folds": positive / len(values),
                "Negative_Frequency_All_Folds": negative / len(values),
                "Zero_Frequency_All_Folds": zero / len(values),
                "Dominant_Nonzero_Sign": dominant_sign,
                "Dominant_Sign_Consistency_Among_Selected":
                    dominant_nonzero_fraction,
            }
        )

    sign_df = pd.DataFrame(records).merge(
        stability[
            [
                "Feature",
                "L1_Selection_Frequency",
                "Spearman_Retention_Frequency",
                "Mean_Coefficient_Across_Outer_Models",
                "Mean_Absolute_Coefficient_Across_Outer_Models",
            ]
        ],
        on="Feature",
        how="left",
    )

    sign_df = sign_df.sort_values(
        [
            "L1_Selection_Frequency",
            "Dominant_Sign_Consistency_Among_Selected",
            "Mean_Absolute_Coefficient_Across_Outer_Models",
        ],
        ascending=False,
    )

    sign_df.to_csv(
        outdir / "Coefficient_Sign_Consistency.csv",
        index=False,
    )

    plot_df = sign_df.head(min(top_n, len(sign_df))).copy()
    plot_df = plot_df.sort_values(
        "L1_Selection_Frequency",
        ascending=True,
    )

    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(
        figsize=(11, max(7, 0.65 * len(plot_df) + 2))
    )

    ax.barh(
        y,
        plot_df["Positive_Frequency_All_Folds"],
        label="Positive coefficient",
    )
    ax.barh(
        y,
        plot_df["Negative_Frequency_All_Folds"],
        left=plot_df["Positive_Frequency_All_Folds"],
        label="Negative coefficient",
    )
    ax.barh(
        y,
        plot_df["Zero_Frequency_All_Folds"],
        left=(
            plot_df["Positive_Frequency_All_Folds"]
            + plot_df["Negative_Frequency_All_Folds"]
        ),
        label="Zero / not selected",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Feature"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of Outer LOOCV Models")
    ax.set_ylabel("Structural Feature")
    ax.set_title(
        f"{analysis_name}: Coefficient Sign Consistency Across Outer Models"
    )
    ax.legend(loc="best")

    fig.tight_layout()

    fig.savefig(
        outdir / "Coefficient_Sign_Consistency.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "Coefficient_Sign_Consistency.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    return sign_df


def save_clone_prediction_diagnostics(
    predictions: pd.DataFrame,
    outdir: Path,
) -> pd.DataFrame:
    """
    Save a clone-level table that highlights confidence and borderline cases.

    Margin_From_Threshold:
        absolute distance from the 0.50 classification threshold.

    Signed_Margin:
        positive -> prediction is on the CR side of 0.50
        negative -> prediction is on the Non-CR side of 0.50
    """
    diag = predictions.copy()

    diag["Signed_Margin_From_0.5"] = (
        diag["Predicted_Probability"] - 0.5
    )
    diag["Absolute_Margin_From_0.5"] = (
        diag["Signed_Margin_From_0.5"].abs()
    )

    diag["Prediction_Confidence_Category"] = pd.cut(
        diag["Absolute_Margin_From_0.5"],
        bins=[-np.inf, 0.05, 0.15, np.inf],
        labels=["Borderline", "Intermediate", "High"],
    )

    diag = diag.sort_values(
        "Absolute_Margin_From_0.5",
        ascending=True,
    )

    diag.to_csv(
        outdir / "Clone_Level_Prediction_Diagnostics.csv",
        index=False,
    )

    return diag


def save_stability_vs_coefficient_plot(
    stability: pd.DataFrame,
    outdir: Path,
    analysis_name: str,
    top_n: int = 15,
) -> None:
    """
    Scatter plot linking model stability and coefficient direction.

    x-axis:
        L1 selection frequency across outer LOOCV models

    y-axis:
        mean standardized coefficient across outer models

    Stable features far from zero are especially useful candidates for
    biological interpretation.
    """
    plot_df = (
        stability
        .sort_values(
            [
                "L1_Selection_Frequency",
                "Mean_Absolute_Coefficient_Across_Outer_Models",
            ],
            ascending=False,
        )
        .head(min(top_n, len(stability)))
        .copy()
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        plot_df["L1_Selection_Frequency"],
        plot_df["Mean_Coefficient_Across_Outer_Models"],
        s=80,
    )

    for _, row in plot_df.iterrows():
        label = (
            str(row["Feature"])
            .replace("AbsDelta_", "")
            .replace("Delta_", "")
        )

        ax.annotate(
            label,
            (
                row["L1_Selection_Frequency"],
                row["Mean_Coefficient_Across_Outer_Models"],
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("L1 Selection Frequency Across Outer LOOCV Models")
    ax.set_ylabel("Mean Standardized L1 Coefficient Across Outer Models")
    ax.set_title(
        f"{analysis_name}: Feature Stability vs Coefficient Direction"
    )

    fig.tight_layout()

    fig.savefig(
        outdir / "Feature_Stability_vs_Coefficient_Direction.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "Feature_Stability_vs_Coefficient_Direction.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def descriptive_spearman(
    X: pd.DataFrame,
    outdir: Path,
    analysis_name: str,
) -> None:
    """
    Save FULL-DATA correlations for descriptive visualization only.

    These correlations are NOT used for predictive feature selection.
    Actual ML filtering happens independently inside each training fold.
    """
    corr = X.corr(method="spearman")
    corr.to_csv(outdir / "Descriptive_Spearman_Correlation.csv")

    high_pairs = []

    features = corr.columns.tolist()

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            rho = corr.loc[features[i], features[j]]

            if abs(rho) >= SPEARMAN_THRESHOLD:
                high_pairs.append(
                    {
                        "Feature_1": features[i],
                        "Feature_2": features[j],
                        "Spearman_Rho": rho,
                    }
                )

    pd.DataFrame(high_pairs).to_csv(
        outdir / "Highly_Correlated_Feature_Pairs.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(18, 16))
    image = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")

    short = [
        c.replace("AbsDelta_", "").replace("Delta_", "")
        for c in corr.columns
    ]

    ax.set_xticks(np.arange(len(short)))
    ax.set_yticks(np.arange(len(short)))
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_title(
        f"{analysis_name}: Descriptive Full-Dataset Spearman Correlation"
    )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho")

    fig.tight_layout()
    fig.savefig(
        outdir / "Full_Data_Spearman_Heatmap.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "Full_Data_Spearman_Heatmap.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def top_feature_interpretation(
    X: pd.DataFrame,
    y: pd.Series,
    clone_ids: pd.Series,
    stability: pd.DataFrame,
    outdir: Path,
    analysis_name: str,
) -> None:
    """
    Descriptive post-hoc interpretation of the most stable L1-selected features.

    This section does NOT feed back into model training.
    """
    ranked = stability.sort_values(
        [
            "L1_Selection_Frequency",
            "Spearman_Retention_Frequency",
            "Mean_Absolute_Coefficient_Across_Outer_Models",
        ],
        ascending=False,
    )

    top_features = ranked.head(min(TOP_N, len(ranked)))["Feature"].tolist()

    records = []

    for rank, feature in enumerate(top_features, start=1):
        non_cr = X.loc[y.values == 0, feature].astype(float)
        cr = X.loc[y.values == 1, feature].astype(float)

        u_stat, p_value = mannwhitneyu(
            cr,
            non_cr,
            alternative="two-sided",
        )

        effect = rank_biserial_from_u(
            u_stat=float(u_stat),
            n0=len(non_cr),
            n1=len(cr),
        )

        records.append(
            {
                "Rank": rank,
                "Feature": feature,
                "NonCR_Median": float(non_cr.median()),
                "CR_Median": float(cr.median()),
                "Mann_Whitney_U": float(u_stat),
                "P_Value": float(p_value),
                "Rank_Biserial_CR_vs_NonCR": float(effect),
            }
        )

    stats_df = pd.DataFrame(records)

    if not stats_df.empty:
        stats_df["BH_FDR"] = benjamini_hochberg(
            stats_df["P_Value"].to_numpy()
        )

    stats_df.to_csv(outdir / "Top10_Feature_Statistics.csv", index=False)

    # Combined boxplots.
    nplots = len(top_features)
    ncols = 2
    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13, max(5, 4.6 * nrows)),
        squeeze=False,
    )

    plot_df = X.copy()
    plot_df[TARGET_COL] = y.values
    plot_df["Clone_ID"] = clone_ids.values

    for rank, (ax, feature) in enumerate(
        zip(axes.flat, top_features),
        start=1,
    ):
        non_cr = plot_df.loc[
            plot_df[TARGET_COL] == 0, feature
        ].astype(float)

        cr = plot_df.loc[
            plot_df[TARGET_COL] == 1, feature
        ].astype(float)

        row = stats_df.loc[stats_df["Feature"] == feature].iloc[0]

        ax.boxplot(
            [non_cr, cr],
            positions=[1, 2],
            widths=0.45,
            showfliers=False,
        )

        rng = np.random.default_rng(RANDOM_SEED + rank)

        for xpos, group_value in [(1, 0), (2, 1)]:
            group = plot_df.loc[
                plot_df[TARGET_COL] == group_value,
                ["Clone_ID", feature],
            ].copy()

            jitter = rng.normal(0, 0.03, size=len(group))
            xpts = xpos + jitter

            ax.scatter(xpts, group[feature].values, s=25, zorder=3)

            for idx, (_, grow) in enumerate(group.iterrows()):
                ax.annotate(
                    str(grow["Clone_ID"]),
                    (xpts[idx], grow[feature]),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7,
                )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Non-CR", "CR"])
        ax.set_title(
            feature.replace("AbsDelta_", "").replace("Delta_", ""),
            fontsize=11,
        )
        ax.text(
            0.5,
            0.98,
            f"p={row['P_Value']:.3g}\nFDR={row['BH_FDR']:.3g}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

    for ax in axes.flat[nplots:]:
        ax.axis("off")

    fig.suptitle(f"{analysis_name}: Top Structural Features", fontsize=19)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(
        outdir / "Top10_Features_Combined.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "Top10_Features_Combined.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    summary = stats_df.merge(
        stability[
            [
                "Feature",
                "L1_Selection_Frequency",
                "Spearman_Retention_Frequency",
                "Mean_Absolute_Coefficient_Across_Outer_Models",
            ]
        ],
        on="Feature",
        how="left",
    )

    summary.to_csv(
        outdir / "Top10_Stability_and_Effect_Size.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        summary["L1_Selection_Frequency"],
        summary["Rank_Biserial_CR_vs_NonCR"],
        s=70,
    )

    for _, row in summary.iterrows():
        ax.annotate(
            str(int(row["Rank"])),
            (
                row["L1_Selection_Frequency"],
                row["Rank_Biserial_CR_vs_NonCR"],
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=10,
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("L1 selection frequency across outer LOOCV models")
    ax.set_ylabel("Rank-biserial effect size\n(+ = higher in CR)")
    ax.set_title(f"{analysis_name}: Stability vs Descriptive Effect")

    fig.tight_layout()
    fig.savefig(
        outdir / "Top10_Stability_vs_Effect_Size.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "Top10_Stability_vs_Effect_Size.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# 7. FINAL FULL-DATA INTERPRETIVE MODEL
# =============================================================================
# This model is NOT used to estimate predictive performance. It is saved for
# feature interpretation and possible future application only.
# =============================================================================

def fit_final_interpretive_model(
    X: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    outdir: Path,
) -> None:
    """
    Fit one final full-data model for interpretation/deployment only.

    IMPORTANT:
    Performance must still be reported from OUTER LOOCV, not from this model.
    """
    min_class_count = int(y.value_counts().min())

    if min_class_count < 2:
        raise ValueError("Not enough minority-class samples for final inner CV.")

    inner_splits = min(INNER_N_SPLITS, min_class_count)

    inner_cv = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=RANDOM_SEED + 999,
    )

    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid={"classifier__C": C_GRID},
        scoring="balanced_accuracy",
        cv=inner_cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )

    search.fit(X, y)

    model = search.best_estimator_

    spearman = model.named_steps["spearman"]
    classifier = model.named_steps["classifier"]

    selected_after_spearman = list(spearman.selected_features_)
    coefs = classifier.coef_[0]

    final_features = pd.DataFrame(
        {
            "Feature": selected_after_spearman,
            "Coefficient": coefs,
            "Absolute_Coefficient": np.abs(coefs),
            "L1_Selected": ~np.isclose(coefs, 0.0, atol=1e-12),
        }
    ).sort_values(
        ["L1_Selected", "Absolute_Coefficient"],
        ascending=[False, False],
    )

    final_features.to_csv(outdir / "Final_Model_Features.csv", index=False)

    joblib.dump(model, outdir / "Final_Model.joblib")

    with open(outdir / "Final_Model_Metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_C": float(search.best_params_["classifier__C"]),
                "best_inner_balanced_accuracy": float(search.best_score_),
                "n_input_features": len(feature_columns),
                "n_spearman_retained": len(selected_after_spearman),
                "n_l1_selected": int(final_features["L1_Selected"].sum()),
                "note": (
                    "This model is fit to all observations for interpretation/"
                    "future application. Report predictive performance from "
                    "outer LOOCV, not from this full-data fit."
                ),
            },
            f,
            indent=2,
        )


# =============================================================================
# 8. CORE NESTED-VALIDATION ANALYSIS
# =============================================================================
# This function performs one complete signed or absolute analysis.
# =============================================================================

def run_analysis(
    df: pd.DataFrame,
    analysis_name: str,
    feature_prefix: str,
    outdir: Path,
) -> dict:
    """Run one complete signed or absolute analysis."""
    outdir.mkdir(parents=True, exist_ok=True)

    X, y, clone_ids, feature_columns = validate_input(
        df,
        feature_prefix=feature_prefix,
    )

    print("\n" + "=" * 90)
    print(f"{analysis_name.upper()} ANALYSIS")
    print("=" * 90)
    print(f"Clones:             {len(df)}")
    print(f"Candidate features: {len(feature_columns)}")
    print("Class counts:")
    print(y.value_counts().sort_index().to_string())

    outer_cv = LeaveOneOut()

    prediction_records = []
    fold_records = []

    spearman_keep_counts = pd.Series(
        0,
        index=feature_columns,
        dtype=int,
    )

    l1_selection_counts = pd.Series(
        0,
        index=feature_columns,
        dtype=int,
    )

    all_outer_coefficient_vectors = []

    # ------------------------------------------------------------------
    # Outer LOOCV
    # ------------------------------------------------------------------
    # One clone is held out at a time.
    #
    # IMPORTANT:
    # Everything that can learn from the data (Spearman filtering, scaling,
    # and C selection) is fit using ONLY the remaining training clones.
    # The held-out clone is touched only once, for final probability/class
    # prediction in that outer fold.
    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        test_clone = clone_ids.iloc[test_idx[0]]

        min_class_count = int(y_train.value_counts().min())

        if min_class_count < 2:
            raise ValueError(
                f"Outer fold {outer_fold}: not enough minority-class "
                "samples for inner stratified CV."
            )

        inner_splits = min(INNER_N_SPLITS, min_class_count)

        inner_cv = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=RANDOM_SEED + outer_fold,
        )

        search = GridSearchCV(
            estimator=build_pipeline(),
            param_grid={"classifier__C": C_GRID},
            scoring="balanced_accuracy",
            cv=inner_cv,
            refit=True,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_C = float(search.best_params_["classifier__C"])
        best_inner_score = float(search.best_score_)

        spearman = best_model.named_steps["spearman"]
        retained = list(spearman.selected_features_)

        for feature in retained:
            spearman_keep_counts.loc[feature] += 1

        classifier = best_model.named_steps["classifier"]
        coefficients = classifier.coef_[0]

        selected_by_l1 = [
            feature
            for feature, coefficient in zip(retained, coefficients)
            if not np.isclose(coefficient, 0.0, atol=1e-12)
        ]

        for feature in selected_by_l1:
            l1_selection_counts.loc[feature] += 1

        coefficient_vector = pd.Series(
            0.0,
            index=feature_columns,
            dtype=float,
        )

        for feature, coefficient in zip(retained, coefficients):
            coefficient_vector.loc[feature] = float(coefficient)

        all_outer_coefficient_vectors.append(coefficient_vector)

        probability = float(best_model.predict_proba(X_test)[0, 1])
        predicted_class = int(probability >= 0.5)
        true_class = int(y_test.iloc[0])

        prediction_records.append(
            {
                "Outer_Fold": outer_fold,
                "Clone_ID": test_clone,
                "Observed_Cross_Reactive": true_class,
                "Predicted_Probability": probability,
                "Predicted_Class": predicted_class,
                "Correct": bool(predicted_class == true_class),
                "Best_C": best_C,
                "Best_Inner_Balanced_Accuracy": best_inner_score,
                "N_Spearman_Retained": len(retained),
                "N_L1_Selected": len(selected_by_l1),
            }
        )

        fold_records.append(
            {
                "Outer_Fold": outer_fold,
                "Held_Out_Clone": test_clone,
                "Best_C": best_C,
                "Best_Inner_Balanced_Accuracy": best_inner_score,
                "N_Candidate_Features": len(feature_columns),
                "N_Spearman_Retained": len(retained),
                "Spearman_Retained_Features": "; ".join(retained),
                "N_L1_Selected": len(selected_by_l1),
                "L1_Selected_Features": "; ".join(selected_by_l1),
                "Spearman_Dropped_Constants": "; ".join(
                    spearman.dropped_constant_features_
                ),
                "Spearman_Dropped_Correlated": "; ".join(
                    spearman.dropped_correlated_features_
                ),
            }
        )

        print(
            f"Fold {outer_fold:02d}/{len(df)} | "
            f"Held out: {test_clone:<10} | "
            f"true={true_class} pred={predicted_class} "
            f"P(CR)={probability:.3f} | "
            f"C={best_C:g} | "
            f"Spearman={len(retained)} | "
            f"L1={len(selected_by_l1)}"
        )

    predictions = pd.DataFrame(prediction_records)
    folds = pd.DataFrame(fold_records)

    predictions.to_csv(
        outdir / "Outer_LOOCV_Predictions.csv",
        index=False,
    )
    folds.to_csv(
        outdir / "Outer_LOOCV_Fold_Details.csv",
        index=False,
    )

    y_true = predictions["Observed_Cross_Reactive"].to_numpy(dtype=int)
    y_prob = predictions["Predicted_Probability"].to_numpy(dtype=float)
    y_pred = predictions["Predicted_Class"].to_numpy(dtype=int)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    auc = roc_auc_score(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    mcc = matthews_corrcoef(y_true, y_pred)

    performance = pd.DataFrame(
        {
            "Metric": [
                "ROC_AUC",
                "Accuracy",
                "Balanced_Accuracy",
                "Sensitivity",
                "Specificity",
                "Precision",
                "F1",
                "Matthews_Correlation_Coefficient",
                "True_Negatives",
                "False_Positives",
                "False_Negatives",
                "True_Positives",
            ],
            "Value": [
                auc,
                accuracy,
                balanced_acc,
                sensitivity,
                specificity,
                precision,
                f1,
                mcc,
                tn,
                fp,
                fn,
                tp,
            ],
        }
    )

    performance.to_csv(
        outdir / "Performance_Metrics.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Bootstrap CI
    # ------------------------------------------------------------------
    auc_low, auc_high = bootstrap_metric_ci(
        y_true,
        y_prob,
        y_pred,
        "ROC_AUC",
        seed=RANDOM_SEED,
    )

    bal_low, bal_high = bootstrap_metric_ci(
        y_true,
        y_prob,
        y_pred,
        "Balanced_Accuracy",
        seed=RANDOM_SEED + 1,
    )

    mcc_low, mcc_high = bootstrap_metric_ci(
        y_true,
        y_prob,
        y_pred,
        "MCC",
        seed=RANDOM_SEED + 2,
    )

    uncertainty = pd.DataFrame(
        {
            "Metric": [
                "ROC_AUC",
                "Balanced_Accuracy",
                "Matthews_Correlation_Coefficient",
            ],
            "Estimate": [auc, balanced_acc, mcc],
            "CI_95_Lower": [auc_low, bal_low, mcc_low],
            "CI_95_Upper": [auc_high, bal_high, mcc_high],
            "Method": [
                "Bootstrap of outer-LOOCV held-out predictions"
            ] * 3,
        }
    )

    uncertainty.to_csv(
        outdir / "Performance_with_Bootstrap_CI.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Feature stability
    # ------------------------------------------------------------------
    coef_matrix = pd.DataFrame(all_outer_coefficient_vectors)
    coef_matrix.to_csv(
        outdir / "Outer_Model_Coefficients.csv",
        index=False,
    )

    mean_coef = coef_matrix.mean(axis=0)
    mean_abs_coef = coef_matrix.abs().mean(axis=0)

    stability = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Spearman_Times_Retained": [
                int(spearman_keep_counts.loc[f]) for f in feature_columns
            ],
            "Spearman_Retention_Frequency": [
                spearman_keep_counts.loc[f] / len(df)
                for f in feature_columns
            ],
            "L1_Times_Selected": [
                int(l1_selection_counts.loc[f]) for f in feature_columns
            ],
            "L1_Selection_Frequency": [
                l1_selection_counts.loc[f] / len(df)
                for f in feature_columns
            ],
            "Mean_Coefficient_Across_Outer_Models": [
                mean_coef.loc[f] for f in feature_columns
            ],
            "Mean_Absolute_Coefficient_Across_Outer_Models": [
                mean_abs_coef.loc[f] for f in feature_columns
            ],
        }
    ).sort_values(
        [
            "L1_Selection_Frequency",
            "Spearman_Retention_Frequency",
            "Mean_Absolute_Coefficient_Across_Outer_Models",
        ],
        ascending=False,
    )

    stability.to_csv(
        outdir / "Feature_Stability.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Figures and descriptive analyses
    # ------------------------------------------------------------------

    # Plot one probability for every completely held-out outer-LOOCV clone.
    # This is the direct visualization of the predictions used to calculate
    # ROC AUC and the threshold-based classification metrics.
    save_heldout_prediction_plot(
        predictions=predictions,
        outdir=outdir,
        analysis_name=analysis_name,
    )

    save_roc(y_true, y_prob, auc, outdir, analysis_name)
    save_confusion(cm, outdir, analysis_name)
    save_feature_stability_plot(stability, outdir, analysis_name)

    # Additional model diagnostics and interpretation plots.
    average_precision = save_precision_recall_curve(
        y_true=y_true,
        y_prob=y_prob,
        outdir=outdir,
        analysis_name=analysis_name,
    )

    save_feature_selection_heatmap(
        coef_matrix=coef_matrix,
        folds=folds,
        stability=stability,
        outdir=outdir,
        analysis_name=analysis_name,
        top_n=20,
    )

    sign_consistency = save_coefficient_sign_consistency(
        coef_matrix=coef_matrix,
        stability=stability,
        outdir=outdir,
        analysis_name=analysis_name,
        top_n=TOP_N,
    )

    clone_diagnostics = save_clone_prediction_diagnostics(
        predictions=predictions,
        outdir=outdir,
    )

    save_stability_vs_coefficient_plot(
        stability=stability,
        outdir=outdir,
        analysis_name=analysis_name,
        top_n=15,
    )

    descriptive_spearman(
        X,
        outdir,
        analysis_name,
    )

    top_feature_interpretation(
        X,
        y,
        clone_ids,
        stability,
        outdir,
        analysis_name,
    )

    fit_final_interpretive_model(
        X,
        y,
        feature_columns,
        outdir,
    )

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    with open(outdir / "Analysis_Summary.txt", "w", encoding="utf-8") as f:
        f.write(f"{analysis_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"N clones: {len(df)}\n")
        f.write(f"Candidate structural predictors: {len(feature_columns)}\n")
        f.write(f"Spearman threshold: {SPEARMAN_THRESHOLD}\n")
        f.write(f"Outer CV: Leave-One-Out CV\n")
        f.write(f"Inner CV max splits: {INNER_N_SPLITS}\n")
        f.write("Classifier: L1 logistic regression, class_weight='balanced'\n")
        f.write("ELISA columns used as predictors: NO\n\n")
        f.write("Outer-LOOCV performance\n")
        f.write("-" * 80 + "\n")
        f.write(performance.to_string(index=False))
        f.write("\n\nBootstrap uncertainty\n")
        f.write("-" * 80 + "\n")
        f.write(uncertainty.to_string(index=False))
        f.write("\n")

    print("\nPerformance:")
    print(performance.to_string(index=False))

    return {
        "Analysis": analysis_name,
        "N_Clones": len(df),
        "N_Candidate_Features": len(feature_columns),
        "ROC_AUC": auc,
        "Average_Precision": average_precision,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1": f1,
        "MCC": mcc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Output_Directory": str(outdir),
    }



def validate_cross_sheet_consistency(
    signed_df: pd.DataFrame,
    absolute_df: pd.DataFrame,
) -> None:
    """
    Require the signed and absolute analyses to use the same clones and targets.
    """
    for name, df in [
        ("ML_Ready_Signed", signed_df),
        ("ML_Ready_Absolute", absolute_df),
    ]:
        for required in [ID_COL, TARGET_COL]:
            if required not in df.columns:
                raise ValueError(
                    f"{name} is missing required column '{required}'."
                )

        if df[ID_COL].duplicated().any():
            raise ValueError(f"{name} contains duplicate Clone_ID values.")

    signed_target = (
        signed_df[[ID_COL, TARGET_COL]]
        .assign(**{ID_COL: lambda d: d[ID_COL].astype(str).str.strip()})
        .set_index(ID_COL)[TARGET_COL]
        .sort_index()
    )

    absolute_target = (
        absolute_df[[ID_COL, TARGET_COL]]
        .assign(**{ID_COL: lambda d: d[ID_COL].astype(str).str.strip()})
        .set_index(ID_COL)[TARGET_COL]
        .sort_index()
    )

    if set(signed_target.index) != set(absolute_target.index):
        missing_from_absolute = sorted(
            set(signed_target.index) - set(absolute_target.index)
        )
        missing_from_signed = sorted(
            set(absolute_target.index) - set(signed_target.index)
        )
        raise ValueError(
            "Signed and absolute sheets contain different clone sets.\n"
            f"Missing from absolute: {missing_from_absolute}\n"
            f"Missing from signed: {missing_from_signed}"
        )

    signed_target = pd.to_numeric(signed_target, errors="raise").astype(int)
    absolute_target = pd.to_numeric(absolute_target, errors="raise").astype(int)

    if not signed_target.equals(absolute_target):
        mismatched = pd.DataFrame(
            {
                "Signed_Target": signed_target,
                "Absolute_Target": absolute_target,
            }
        )
        mismatched = mismatched.loc[
            mismatched["Signed_Target"] != mismatched["Absolute_Target"]
        ]
        raise ValueError(
            "Cross_Reactive_Target differs between signed and absolute sheets:\n"
            + mismatched.to_string()
        )


# =============================================================================
# 9. COMMAND-LINE ENTRY POINT
# =============================================================================
# Load both ML-ready sheets, verify cross-sheet consistency, run both models,
# and save a side-by-side model comparison.
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe L1 logistic regression on signed and absolute "
            "structural delta feature tables."
        )
    )

    parser.add_argument(
        "--input-workbook",
        required=True,
        type=Path,
        help="ML-ready workbook produced by script 09.",
    )

    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Root output directory. Signed/ and Absolute/ folders are created.",
    )

    args = parser.parse_args()

    if not args.input_workbook.exists():
        raise FileNotFoundError(
            f"Input workbook not found:\n{args.input_workbook}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)

    signed_df = pd.read_excel(
        args.input_workbook,
        sheet_name="ML_Ready_Signed",
    )

    absolute_df = pd.read_excel(
        args.input_workbook,
        sheet_name="ML_Ready_Absolute",
    )

    validate_cross_sheet_consistency(
        signed_df=signed_df,
        absolute_df=absolute_df,
    )

    print("\nInput consistency check passed:")
    print("  Signed and absolute clone sets match.")
    print("  Cross_Reactive_Target matches clone-by-clone.")

    signed_result = run_analysis(
        df=signed_df,
        analysis_name="Signed Delta",
        feature_prefix="Delta_",
        outdir=args.output_root / "Signed_Delta",
    )

    absolute_result = run_analysis(
        df=absolute_df,
        analysis_name="Absolute Delta",
        feature_prefix="AbsDelta_",
        outdir=args.output_root / "Absolute_Delta",
    )

    comparison = pd.DataFrame(
        [signed_result, absolute_result]
    )

    comparison.to_csv(
        args.output_root / "Model_Comparison.csv",
        index=False,
    )

    comparison.to_excel(
        args.output_root / "Model_Comparison.xlsx",
        index=False,
    )

    print("\n" + "=" * 90)
    print("BOTH ANALYSES COMPLETE")
    print("=" * 90)
    print(comparison[
        [
            "Analysis",
            "ROC_AUC",
            "Average_Precision",
            "Balanced_Accuracy",
            "MCC",
            "Sensitivity",
            "Specificity",
        ]
    ].to_string(index=False))

    print(f"\nAll outputs saved under:\n{args.output_root}")
    print("\nSigned outputs:")
    print(args.output_root / "Signed_Delta")
    print("\nAbsolute outputs:")
    print(args.output_root / "Absolute_Delta")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR")
        print("=" * 90)
        print(exc)
        sys.exit(1)
