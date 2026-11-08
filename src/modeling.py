"""
Validation and modelling utilities: GroupKFold by user, F1-optimal threshold.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score

SEED = 42


def best_threshold(y, p):
    """Threshold maximising pooled F1 on out-of-fold probabilities."""
    grid = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y, p >= t) for t in grid]
    i = int(np.argmax(scores))
    return grid[i], scores[i]


def cv_oof(make_model, X, y, groups, n_splits=5, fit_kwargs=None, seed=SEED):
    """Out-of-fold probabilities with folds grouped by user (no user in train and val)."""
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    fold_auc = []
    for tr_idx, va_idx in folds.split(X, y, groups):
        m = make_model()
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], **(fit_kwargs or {}))
        oof[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
        fold_auc.append(roc_auc_score(y.iloc[va_idx], oof[va_idx]))
    return oof, fold_auc


def summarize(name, y, oof, fold_auc):
    t, f1 = best_threshold(y, oof)
    return {"model": name, "cv_auc": np.mean(fold_auc), "cv_auc_std": np.std(fold_auc),
            "best_threshold": t, "cv_f1": f1}
