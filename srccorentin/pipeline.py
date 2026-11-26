from .data_loading import json_to_dataframe, get_account_status, get_artist_genres
from .music_features import (prepare_music_columns, build_user_features, build_artist_features,
                             add_music_features, add_genre_share)
from .recency_features import add_days_ago, build_user_recency, build_artist_recency, add_recency_features
from .csv_features import add_csv_features
from .behaviour_features import add_ranking_features, add_behaviour_features


def build_features(train, test, data):
    """Run every feature engineering step, in the same order as the notebook.
    Return train_fe and test_fe."""

    # 1. Listening data from the JSON: one row = one track played
    df_music = prepare_music_columns(json_to_dataframe(data))
    df_music = add_days_ago(df_music, data["history_ends_before"])

    # 2. Tables computed once from the JSON
    status = get_account_status(data)
    artist_genres = get_artist_genres(data)
    df_activity = build_user_features(df_music)
    df_artist = build_artist_features(df_music)
    df_recency = build_user_recency(df_music)
    df_artist_recency = build_artist_recency(df_music)

    # 3. Apply exactly the same steps to train and test
    def transform(df):
        df = add_music_features(df, status, df_activity, df_artist)
        df = add_genre_share(df, artist_genres)
        df = add_csv_features(df)
        df = add_recency_features(df, df_recency, df_artist_recency)
        df = add_ranking_features(df)
        df = add_behaviour_features(df)
        return df

    train_fe, test_fe = transform(train), transform(test)

    # Verify that merging did not create additional rows.
    assert len(train_fe) == len(train)
    assert len(test_fe) == len(test)

    return train_fe, test_fe
