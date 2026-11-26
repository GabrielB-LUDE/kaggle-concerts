import numpy as np
import pandas as pd


def prepare_music_columns(df_music):
    """PREPARE NEW COLUMNS IN THE MUSIC DATAFRAME
    df_music: one row = one track played by a user"""
    df_music = df_music.copy()

    # Calculate the proportion of each track listened to.
    # Example: 90 seconds played / 180 seconds = 0.5
    # clip(0, 1) keeps the result between 0 and 1.
    df_music["completion"] = (df_music["played_ms"] / df_music["track_ms"]).clip(0, 1)

    # Create a binary column indicating whether the user started the track through a search.
    # 1 = search, 0 = other source
    df_music["searched"] = (df_music["source"] == "search").astype(int)

    return df_music


def build_user_features(df_music):
    """CREATE USER-LEVEL FEATURES
    df_activity: one row = one user.
    df_music must already contain the 'completion' and 'searched' columns."""

    # Group all tracks belonging to the same user.
    # Store the grouping in g to avoid repeating groupby.
    g = df_music.groupby("music_app_user_id")

    # A. Total plays and sessions
    # count number of tracks per user / count sessions per user
    df_activity = pd.concat([g.size(), g["session_id"].nunique()], axis=1)
    # name the columns
    df_activity.columns = ["total_plays", "total_sessions"]

    # Add all milliseconds listened to by each user.
    # Divide by 3,600,000 to convert milliseconds into hours.
    df_activity["listening_hours"] = g["played_ms"].sum() / 3600000

    # Calculate the average proportion of a track listened to.
    # Example: 0.8 means the user listens to 80% of a track on average.
    df_activity["completion_rate"] = g["completion"].mean()

    # Calculate the proportion of play events marked as saved.
    # True = 1 and False = 0, so the mean gives a proportion.
    df_activity["save_rate"] = g["saved"].mean()

    # Calculate the proportion of tracks started through search.
    # Example: 0.3 means 30% of plays came from search.
    df_activity["search_rate"] = g["searched"].mean()

    # Count the number of DIFFERENT artists listened to.
    # nunique() does not count duplicates.
    df_activity["unique_artists"] = g["artist_id"].nunique()

    # Count the number of DIFFERENT tracks listened to.
    # Listening to the same song 5 times counts as 1.
    df_activity["unique_tracks"] = g["track_id"].nunique()

    # Calculate the average number of tracks per session.
    df_activity["plays_per_session"] = df_activity["total_plays"] / df_activity["total_sessions"]

    # Add the genre counts and proportions.
    return add_genre_features(df_activity, df_music)


def add_genre_features(df_activity, df_music):
    """Add the number and proportion of plays per genre for each user."""

    # Extract all the different genres present in df_music.
    # Example: ["pop", "metal", "alternative", ...]
    genres = df_music["genre"].dropna().unique()

    # Store the number of plays per user for each genre.
    genre_counts = {}
    for genre in genres:
        # Keep only tracks of the current genre, then count the tracks per user
        filtered_tracks = df_music[df_music["genre"] == genre]
        genre_counts[genre] = filtered_tracks.groupby("music_app_user_id").size()

    # Transform the dictionary into a DataFrame.
    # One row = one user. One column = one genre.
    # Replace missing genre counts with 0.
    # NaN here means the user never played this genre.
    df_genres = pd.DataFrame(genre_counts).reindex(df_activity.index).fillna(0).astype(int)

    # Combine the general listening features with genre counts.
    # Pandas matches users using music_app_user_id as the index.
    df_activity = pd.concat([df_activity, df_genres], axis=1)

    # Calculate the proportion of plays for each genre.
    # Example: 20 pop plays / 100 total plays = 0.2.
    for genre in genres:
        df_activity[f"{genre}_pct"] = df_activity[genre] / df_activity["total_plays"]

    return df_activity


def build_artist_features(df_music):
    """Create artist-specific features.
    Each row represents a combination of one user and one artist."""

    # Group the music data by the user ID AND the artist ID.
    # Each group contains all the tracks played
    # by a particular user for a particular artist.
    g_artist = df_music.groupby(["music_app_user_id", "artist_id"])

    df_artist = pd.DataFrame()
    # Count the number of plays per user and artist.
    df_artist["artist_plays"] = g_artist.size()
    # Calculate the total listening time in hours.
    df_artist["artist_listening_hours"] = g_artist["played_ms"].sum() / 3600000
    # Calculate the average track completion rate.
    df_artist["artist_completion"] = g_artist["completion"].mean()
    # Count the plays marked as saved.
    df_artist["artist_saved"] = g_artist["saved"].sum()
    # Count the plays initiated through search.
    df_artist["artist_searches"] = g_artist["searched"].sum()
    # Count the number of different tracks played.
    df_artist["artist_unique_tracks"] = g_artist["track_id"].nunique()
    # Count the number of different listening sessions.
    df_artist["artist_sessions"] = g_artist["session_id"].nunique()

    # Transform the two indexes into normal columns.
    # This will make merging with the CSV easier.
    return df_artist.reset_index()


def add_music_features(df, status, df_activity, df_artist):
    """Add music features to a CSV DataFrame (train or test)."""

    # Create a copy to avoid modifying the original CSV.
    df = df.copy()

    # Retrieve the music account status for each user.
    df["app_status"] = df["music_app_user_id"].map(status)

    # Identify users who have not linked a music account.
    df.loc[df["music_app_user_id"].isna(), "app_status"] = "unlinked"

    # Merge general listening features.
    # Match the user ID in the CSV with the index of df_activity.
    # how="left" keeps every original CSV row.
    df = df.merge(df_activity, left_on="music_app_user_id", right_index=True,
                  how="left", validate="many_to_one")

    # Merge artist-specific listening features.
    # Match BOTH the user ID and the concert artist ID.
    # This tells us how much the user listens to
    # the artist of the concert they are offered.
    df = df.merge(df_artist, on=["music_app_user_id", "artist_id"],
                  how="left", validate="many_to_one")

    # Replace missing values with 0 ONLY for active users:
    # for them, a missing value means zero activity.
    # Missing listening histories remain NaN.
    active = df["app_status"] == "active"
    count_cols = ["artist_plays", "artist_listening_hours", "artist_saved",
                  "artist_searches", "artist_unique_tracks", "artist_sessions"]
    df.loc[active, count_cols] = df.loc[active, count_cols].fillna(0)

    # Calculate the proportion of all plays belonging to the concert artist.
    # Example: 30 Taylor Swift plays / 100 total plays = 0.3
    df["artist_play_pct"] = df["artist_plays"] / df["total_plays"]

    # Calculate the proportion of listening time dedicated to the concert artist.
    df["artist_hours_pct"] = df["artist_listening_hours"] / df["listening_hours"].replace(0, np.nan)

    return df


def genre_compatibility(row, artist_genres):
    """Receive ONE row of the CSV (one user and one concert) and return
    the share of the user's listening that matches the concert genres."""

    # If listening history is unavailable,
    # we cannot calculate the user's preferences.
    if pd.isna(row["total_plays"]):
        return np.nan

    # If the artist is unknown, return a missing value.
    artist_id = row["artist_id"]
    if artist_id not in artist_genres:
        return np.nan

    # Add the user's listening proportion for each genre of the artist.
    # Example: "pop" becomes "pop_pct".
    score = 0
    for genre in set(artist_genres[artist_id]):
        score += row.get(f"{genre}_pct", 0)

    return score


def add_genre_share(df, artist_genres):
    """Apply genre_compatibility to every row (axis=1 = row by row)."""
    df = df.copy()
    df["concert_genre_share"] = df.apply(genre_compatibility, axis=1, artist_genres=artist_genres)
    return df
