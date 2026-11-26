import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def optimize_threshold(y_true, probabilities):
    """Test thresholds from 0.10 to 0.90 and keep the one with the best F1."""
    best_f1, best_threshold = 0, 0.5

    for threshold in np.arange(0.1, 0.91, 0.01):
        # Convert probabilities into binary predictions, then calculate F1.
        score = f1_score(y_true, (probabilities >= threshold).astype(int))
        # Keep the best combination.
        if score > best_f1:
            best_f1, best_threshold = score, threshold

    return best_threshold, best_f1


def evaluate(model, X_valid, y_valid, name="Model"):
    """Print Accuracy, ROC-AUC, F1 (threshold 0.5) and the optimised F1.
    Return the metrics in a dictionary."""

    # Predict 0 or 1, then the probability of buying a ticket.
    # [:, 1] selects the probability of class 1.
    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]

    threshold, best_f1 = optimize_threshold(y_valid, y_proba)

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_valid, y_pred),
        "ROC-AUC": float(roc_auc_score(y_valid, y_proba)),
        "F1 (0.5)": f1_score(y_valid, y_pred),
        "Best threshold": round(float(threshold), 2),
        "Best F1": best_f1,
    }
    print(results)
    return results


def plot_feature_importance(model_rf, top=20):
    """Display the most important features of a tree-based pipeline."""

    # Column names after preprocessing and importance of each feature.
    feature_names = model_rf.named_steps["preprocessing"].get_feature_names_out()
    importance = model_rf.named_steps["model"].feature_importances_

    pd.Series(importance, index=feature_names).nlargest(top).sort_values().plot(
        kind="barh", figsize=(10, 8))
    plt.title(f"Top {top} Features")
    plt.xlabel("Feature importance")
    plt.show()


def plot_lr_coefficients(model_lr, top=20):
    """Display the largest Logistic Regression coefficients (absolute value),
    keeping their original positive / negative sign."""
    feature_names = model_lr.named_steps["preprocessing"].get_feature_names_out()
    coefficients = pd.Series(model_lr.named_steps["model"].coef_[0], index=feature_names)

    top_features = coefficients.abs().nlargest(top).index
    coefficients.loc[top_features].sort_values().plot(kind="barh", figsize=(10, 8))
    plt.axvline(0, color="black")
    plt.title("Logistic Regression - Feature Coefficients")
    plt.show()
