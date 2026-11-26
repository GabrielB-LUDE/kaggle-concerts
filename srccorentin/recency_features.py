import pandas as pd


def add_days_ago(df_music, history_end):
    """Calculate how many days ago each track was played.
    history_end: reference date = end of the listening history."""
    df_music = df_music.copy()

    # Convert the listening date into datetime format.
    df_music["played_at"] = pd.to_datetime(df_music["played_at"], utc=True)

    snapshot = pd.Timestamp(history_end)
    df_music["days_ago"] = (snapshot - df_music["played_at"]).dt.total_seconds() / 86400

    return df_music


def build_user_recency(df_music):
    """Recency features per user (df_music must contain 'days_ago')."""
    g = df_music.groupby("music_app_user_id")
    df_recency = pd.DataFrame()

    # Number of days since the user's last listening event.
    # min() retrieves the most recent event.
    df_recency["days_since_last_play"] = g["days_ago"].min()

    # Count the number of plays during the last 7 and 30 days.
    music_7d = df_music[df_music["days_ago"].between(0, 7)]
    music_30d = df_music[df_music["days_ago"].between(0, 30)]
    df_recency["plays_7d"] = music_7d.groupby("music_app_user_id").size()
    df_recency["plays_30d"] = music_30d.groupby("music_app_user_id").size()

    # NaN means the active user had no plays in that period.
    return df_recency.fillna(0)


def build_artist_recency(df_music):
    """Recency features per user and artist."""
    keys = ["music_app_user_id", "artist_id"]
    g_artist = df_music.groupby(keys)
    df_artist_recency = pd.DataFrame()

    # Number of days since the user last listened to this artist.
    df_artist_recency["artist_days_since_last_play"] = g_artist["days_ago"].min()

    # Number of plays of this artist in the last 7 and 30 days.
    music_7d = df_music[df_music["days_ago"].between(0, 7)]
    music_30d = df_music[df_music["days_ago"].between(0, 30)]
    df_artist_recency["artist_plays_7d"] = music_7d.groupby(keys).size()
    df_artist_recency["artist_plays_30d"] = music_30d.groupby(keys).size()

    # Replace missing recent play counts with zero.
    # Transform the indexes into normal columns for merging.
    return df_artist_recency.fillna(0).reset_index()


def add_recency_features(df, df_recency, df_artist_recency):
    """Merge the recency features into a CSV DataFrame."""
    df = df.copy()

    # Merge general listening recency using the user ID.
    df = df.merge(df_recency, left_on="music_app_user_id", right_index=True,
                  how="left", validate="many_to_one")

    # Merge artist recency using both user and artist IDs.
    df = df.merge(df_artist_recency, on=["music_app_user_id", "artist_id"],
                  how="left", validate="many_to_one")

    # For active users, no plays of an artist means zero.
    active = df["app_status"] == "active"
    columns = ["artist_plays_7d", "artist_plays_30d"]
    df.loc[active, columns] = df.loc[active, columns].fillna(0)

    return df
