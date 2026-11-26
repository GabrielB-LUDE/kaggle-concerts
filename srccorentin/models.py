from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_random_forest(preprocessor):
    """Preprocessing + Random Forest.
    The data is prepared automatically before being given to the model."""
    rf = RandomForestClassifier(
        n_estimators=200,      # Number of decision trees.
        max_depth=12,          # Maximum depth of each decision tree.
        min_samples_leaf=5,    # Minimum number of observations in a leaf.
        random_state=42,       # Obtain reproducible results.
        n_jobs=-1              # Use all available CPU cores.
    )
    return Pipeline([("preprocessing", clone(preprocessor)), ("model", rf)])


def build_logistic_regression(preprocessor):
    """Preprocessing + scaling + Logistic Regression."""
    return Pipeline([
        # Same preprocessing as Random Forest: missing values and one-hot encoding.
        ("preprocessing", clone(preprocessor)),
        # Standardize the features for Logistic Regression.
        # with_mean=False supports sparse one-hot encoded data.
        ("scaling", StandardScaler(with_mean=False)),
        # max_iter allows more iterations to find the optimal coefficients.
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ])


def build_lightgbm(preprocessor):
    """Preprocessing + LightGBM (pip install lightgbm)."""
    from lightgbm import LGBMClassifier

    lgbm = LGBMClassifier(
        n_estimators=300,        # Number of successive boosting trees.
        learning_rate=0.05,      # How much each new tree contributes.
        num_leaves=31,           # Maximum number of leaves per tree.
        max_depth=6,             # Maximum depth of each tree.
        min_child_samples=20,    # Minimum observations per leaf.
        subsample=0.8,           # Randomly select 80% of rows during boosting.
        subsample_freq=1,
        colsample_bytree=0.8,    # Randomly select 80% of features per tree.
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    return Pipeline([("preprocessing", clone(preprocessor)), ("model", lgbm)])
