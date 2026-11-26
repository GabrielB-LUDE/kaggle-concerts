import json
import pandas as pd


def load_data(train_path="train.csv", test_path="test.csv", json_path="music_app.json"):
    """Read the training CSV, the testing CSV and the music app JSON."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    return train, test, data


def json_to_dataframe(data):
    """Transform JSON into dataframe.
    One row = one track played by a user."""
    rows = []
    for account in data["accounts"]:
        for session in account["sessions"]:
            for track in session["tracks"]:
                rows.append({"music_app_user_id": account["music_app_user_id"],
                             "session_id": session["session_id"],
                             **track})

    return pd.DataFrame(rows)


def get_account_status(data):
    """Return a dictionary {music_app_user_id: activity_status}."""
    status = {}

    # Go through each account in the JSON data
    for account in data["accounts"]:
        # Add the user ID as the key
        # Add the activity status as the value
        status[account["music_app_user_id"]] = account["activity_status"]

    return status


def get_artist_genres(data):
    """Return a dictionary {artist_id: list of genres}."""
    artist_genres = {}

    # Go through every artist in the JSON catalog.
    for artist in data["catalog"]["artists"]:
        # Store the artist ID as the key
        # and the list of genres as the value.
        artist_genres[artist["artist_id"]] = artist["genres"]

    return artist_genres
