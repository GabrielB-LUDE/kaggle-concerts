import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Columns we don't want to give to the model.
# IDs and raw dates are removed.
# We keep the numerical features created from the dates.
DROP_COLUMNS = ["buy", "row_id", "user_id", "music_app_user_id", "artist_id", "concert_id",
                "concert_at", "sale_opened_at", "prediction_at", "travel_search_arrival"]


def encode_genres(df):
    """Transform the genres column.
    Example: "pop|alternative" becomes two binary columns."""
    df = df.copy()

    # One binary column per concert genre, with a prefix to distinguish
    # concert genres from the user's listening genres.
    concert_genres = df["genres"].str.get_dummies(sep="|").add_prefix("concert_genre_")

    # Add the new columns and remove the original text column.
    return pd.concat([df, concert_genres], axis=1).drop(columns="genres")


def prepare_X_y(train_fe, test_fe):
    """Return X, y, groups (user_id) and X_test ready for the models."""

    # y contains what we want to predict. 1 = bought a ticket, 0 = did not buy.
    y = train_fe["buy"]

    # Keep user_id separately for the train/validation split.
    # We don't want the same user in both datasets.
    groups = train_fe["user_id"]

    # X contains all the features used to predict buy.
    X = encode_genres(train_fe.drop(columns=DROP_COLUMNS, errors="ignore"))
    X_test = encode_genres(test_fe.drop(columns=DROP_COLUMNS, errors="ignore"))

    # Ensure the test data has exactly the same feature columns.
    X_test = X_test.reindex(columns=X.columns, fill_value=0)

    # Check for target leakage
    assert "buy" not in X.columns

    return X, y, groups, X_test


def split_by_user(X, y, groups, test_size=0.2, random_state=42):
    """Keep 80% of users for training and 20% for validation.
    groups ensures that the same user cannot appear in both datasets."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, valid_idx = next(splitter.split(X, y, groups))

    X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    return X_train, X_valid, y_train, y_valid


def build_preprocessor(X_train):
    """Define how numerical and categorical columns should be prepared."""

    # Identify numerical columns (price, distance, total_plays...)
    # and non-numerical columns (education, city, app_status...).
    num_cols = X_train.select_dtypes(include="number").columns
    cat_cols = X_train.select_dtypes(exclude="number").columns

    return ColumnTransformer([
        # NUMERICAL COLUMNS
        # Replace missing values with the training median.
        # Add indicators to identify missing information.
        ("numerical", SimpleImputer(strategy="median", add_indicator=True), num_cols),

        # CATEGORICAL COLUMNS
        # Replace missing values with "Unknown", then apply one-hot encoding.
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])
