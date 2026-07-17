"""
Customer Churn Prediction with scikit-learn

Binary classification project using scikit-learn's LogisticRegression.
Includes EDA, preprocessing, class-imbalance handling, threshold optimisation,
and full model evaluation.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
)
from sklearn.pipeline import Pipeline


__all__ = [
    "load_data",
    "plot_eda",
    "build_pipeline",
    "find_best_threshold",
    "assign_risk_band",
    "evaluate_model",
]

# -----------------------------
# Logging
# -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "fintech_churn_synthetic.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "age",
    "tenure_years",
    "account_balance",
    "credit_score",
    "num_products",
    "is_active",
    "estimated_salary",
    "complaints",
    "support_tickets",
    "mobile_logins_30d",
    "overdraft_count_12m",
]
TARGET = "churn"

sns.set_theme(style="whitegrid", context="notebook")


# -----------------------------
# Data Loading
# -----------------------------

def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the churn dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Run `python src/generate_synthetic_churn_data.py` first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d rows  |  Churn rate: %.2f%%", len(df), df[TARGET].mean() * 100)
    return df


# -----------------------------
# EDA
# -----------------------------

def save_plot(filename: str) -> None:
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", path)


def plot_eda(df: pd.DataFrame) -> None:
    """Generate exploratory data analysis charts."""
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=TARGET, color="steelblue")
    plt.title("Customer Churn Count")
    plt.xlabel("Churned")
    plt.ylabel("Customers")
    plt.xticks([0, 1], ["Stayed", "Churned"])
    save_plot("01_churn_count.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET, y="tenure_years")
    plt.title("Tenure by Churn Status")
    plt.xlabel("Churned")
    plt.ylabel("Tenure in Years")
    plt.xticks([0, 1], ["Stayed", "Churned"])
    save_plot("02_tenure_by_churn.png")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="complaints", y=TARGET, estimator=np.mean)
    plt.title("Average Churn Rate by Number of Complaints")
    plt.xlabel("Complaints")
    plt.ylabel("Average Churn Rate")
    save_plot("03_churn_by_complaints.png")

    # Correlation heatmap of features
    plt.figure(figsize=(10, 8))
    corr = df[FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    save_plot("04_correlation_heatmap.png")


# -----------------------------
# Model
# -----------------------------

def build_pipeline(class_weight: str = "balanced") -> Pipeline:
    """
    Build a scikit-learn Pipeline: StandardScaler → LogisticRegression.

    class_weight='balanced' automatically adjusts for class imbalance
    by weighting the minority class inversely proportional to its frequency.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )),
    ])


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Use the precision-recall curve to find the threshold that maximises F1.

    This is more principled than a brute-force scan because sklearn's
    precision_recall_curve already evaluates every unique probability value.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one more value than thresholds
    f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    logger.info(
        "Best threshold: %.2f  →  F1=%.3f  Precision=%.3f  Recall=%.3f",
        best_threshold, f1_scores[best_idx], precision[best_idx], recall[best_idx],
    )
    return best_threshold


# -----------------------------
# Risk Bands
# -----------------------------

def assign_risk_band(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.60:
        return "Medium Risk"
    return "High Risk"


# -----------------------------
# Evaluation
# -----------------------------

def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Run full evaluation and save all metric plots."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["Stayed", "Churned"])
    cm = confusion_matrix(y_test, y_pred)

    logger.info("ROC-AUC: %.3f", auc)
    logger.info("Classification Report:\n%s", report)

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Stayed", "Predicted Churned"],
        yticklabels=["Actual Stayed", "Actual Churned"],
    )
    plt.title("Confusion Matrix")
    save_plot("05_confusion_matrix.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"Logistic Regression AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    save_plot("06_roc_curve.png")

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision)
    plt.axvline(x=threshold, linestyle="--", color="red", label=f"Threshold = {threshold:.2f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    save_plot("07_precision_recall_curve.png")

    # Feature importance (model coefficients)
    model = pipeline.named_steps["model"]
    importance = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": model.coef_[0],
        "absolute_importance": np.abs(model.coef_[0]),
    }).sort_values("absolute_importance", ascending=False)

    plt.figure(figsize=(10, 6))
    colors = ["tomato" if c > 0 else "steelblue" for c in importance["coefficient"]]
    importance["color"] = colors
    sns.barplot(data=importance, x="coefficient", y="feature", hue="feature", palette=dict(zip(importance["feature"], colors)), legend=False)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Feature Coefficients\n(positive = increases churn risk, negative = reduces it)")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    save_plot("08_feature_coefficients.png")

    return {
        "auc": auc,
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importance": importance,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    logger.info("Starting Customer Churn Prediction (scikit-learn)")

    # Load
    df = load_data()

    # EDA
    plot_eda(df)

    # Split
    X = df[FEATURES]
    y = df[TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(
        "Train: %d rows  |  Test: %d rows  |  Train churn rate: %.2f%%",
        len(X_train), len(X_test), y_train.mean() * 100,
    )

    # Build and train
    pipeline = build_pipeline(class_weight="balanced")
    pipeline.fit(X_train, y_train)

    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    logger.info("5-Fold CV ROC-AUC: %.3f ± %.3f", cv_auc.mean(), cv_auc.std())

    # Threshold optimisation
    y_prob_train = pipeline.predict_proba(X_train)[:, 1]
    best_threshold = find_best_threshold(y_train, y_prob_train)

    # Evaluate on test set
    results = evaluate_model(pipeline, X_test, y_test, threshold=best_threshold)

    # Save predictions
    test_results = pd.DataFrame({
        "actual_churn": y_test,
        "predicted_churn_probability": results["y_prob"],
        "predicted_churn": results["y_pred"],
    })
    test_results["risk_band"] = test_results["predicted_churn_probability"].apply(assign_risk_band)

    output_path = PROJECT_ROOT / "outputs" / "churn_predictions.csv"
    test_results.to_csv(output_path, index=False)
    logger.info("Saved predictions: %s", output_path)
    logger.info("Risk Band Distribution:\n%s", test_results["risk_band"].value_counts().to_string())

    logger.info("Done.")


if __name__ == "__main__":
    main()
