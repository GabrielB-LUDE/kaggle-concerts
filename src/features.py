"""
Feature engineering for the "Front Row: Who Buys the Ticket?" challenge (ML DSBA M2).

Two families of features:
  1. Tabular features built from train.csv / test.csv (profile, offer, engagement).
  2. Listening features built from music_app.json (user overall + user x concert-artist
     + user x concert-genres).

Usage:
    from features import build_features
    train_df, test_df = build_features("data/")
"""
import json
import numpy as np
import pandas as pd

SNAPSHOT = pd.Timestamp("2026-10-01T00:00:00Z")
CONCERT_ARTISTS = ["A01", "A02", "A03", "A04", "A05"]


# ----------------------------------------------------------------------------
# 1. Music-app JSON -> flat table of track events
# ----------------------------------------------------------------------------
def load_music(path):
    """Flatten the nested JSON into one row per track event.

    Returns (events DataFrame, account status Series, artist->genres dict)."""
    with open(path) as f:
        d = json.load(f)

    artist_genres = {a["artist_id"]: a["genres"] for a in d["catalog"]["artists"]}
    status = {}
    rows = []
    for acc in d["accounts"]:
        uid = acc["music_app_user_id"]
        status[uid] = acc["activity_status"]
        for s in acc["sessions"]:
            sid = s["session_id"]
            for t in s["tracks"]:
                rows.append((uid, sid, t["played_at"], t["artist_id"], t["genre"],
                             t["track_id"], t["played_ms"], t["track_ms"],
                             t["source"] == "search", bool(t["saved"])))
    ev = pd.DataFrame(rows, columns=["mid", "session_id", "played_at", "artist_id", "genre",
                                     "track_id", "played_ms", "track_ms", "searched", "saved"])
    ev["played_at"] = pd.to_datetime(ev["played_at"], utc=True)
    ev["days_ago"] = (SNAPSHOT - ev["played_at"]).dt.total_seconds() / 86400
    ev["completion"] = (ev["played_ms"] / ev["track_ms"]).clip(0, 1)
    ev["completed"] = ev["completion"] >= 0.9
    ev["skipped"] = ev["completion"] < 0.3
    return ev, pd.Series(status, name="app_status"), artist_genres


def user_level_music(ev):
    """Overall listening behaviour of each account (independent of the concert)."""
    g = ev.groupby("mid")
    out = pd.DataFrame({
        "u_n_tracks": g.size(),
        "u_n_sessions": g["session_id"].nunique(),
        "u_listen_hours": g["played_ms"].sum() / 3.6e6,
        "u_search_rate": g["searched"].mean(),
        "u_save_rate": g["saved"].mean(),
        "u_completion": g["completion"].mean(),
        "u_days_since_last": g["days_ago"].min(),
        "u_span_days": g["days_ago"].max() - g["days_ago"].min(),
        "u_n_artists": g["artist_id"].nunique(),
        "u_tracks_7d": ev[ev.days_ago <= 7].groupby("mid").size(),
        "u_tracks_30d": ev[ev.days_ago <= 30].groupby("mid").size(),
    })
    out[["u_tracks_7d", "u_tracks_30d"]] = out[["u_tracks_7d", "u_tracks_30d"]].fillna(0)
    out["u_tracks_per_session"] = out["u_n_tracks"] / out["u_n_sessions"]
    # How many of the 5 concert artists does the user listen to at all?
    ca = ev[ev.artist_id.isin(CONCERT_ARTISTS)]
    out["u_n_concert_artists"] = ca.groupby("mid")["artist_id"].nunique()
    out["u_n_concert_artists"] = out["u_n_concert_artists"].fillna(0)
    return out


def user_artist_music(ev):
    """Affinity of each account for each of the 5 concert artists."""
    ca = ev[ev.artist_id.isin(CONCERT_ARTISTS)]
    g = ca.groupby(["mid", "artist_id"])
    out = pd.DataFrame({
        "a_plays": g.size(),
        "a_hours": g["played_ms"].sum() / 3.6e6,
        "a_search_plays": g["searched"].sum(),
        "a_saved": g["saved"].sum(),
        "a_completion": g["completion"].mean(),
        "a_completed_plays": g["completed"].sum(),
        "a_skip_rate": g["skipped"].mean(),
        "a_distinct_tracks": g["track_id"].nunique(),
        "a_sessions": g["session_id"].nunique(),
        "a_days_since_last": g["days_ago"].min(),
        "a_days_since_first": g["days_ago"].max(),
        "a_plays_7d": ca[ca.days_ago <= 7].groupby(["mid", "artist_id"]).size(),
        "a_plays_30d": ca[ca.days_ago <= 30].groupby(["mid", "artist_id"]).size(),
        "a_search_plays_30d": ca[(ca.days_ago <= 30) & ca.searched].groupby(["mid", "artist_id"]).size(),
    })
    cnt_cols = ["a_plays_7d", "a_plays_30d", "a_search_plays_30d"]
    out[cnt_cols] = out[cnt_cols].fillna(0)
    out["a_search_rate"] = out["a_search_plays"] / out["a_plays"]
    out["a_save_rate"] = out["a_saved"] / out["a_plays"]
    # Recency-weighted engagement (half-life 30 days); search & save weigh more.
    w = np.exp(-ca["days_ago"] / 30) * (1 + ca["searched"] + 2 * ca["saved"]) * ca["completion"]
    out["a_weighted_score"] = w.groupby([ca["mid"], ca["artist_id"]]).sum()
    return out.reset_index()


def user_genre_shares(ev):
    """Share of listening time per genre for each account (wide table)."""
    ms = ev.pivot_table(index="mid", columns="genre", values="played_ms", aggfunc="sum", fill_value=0)
    return ms.div(ms.sum(axis=1), axis=0)


# ----------------------------------------------------------------------------
# 2. Tabular features + merge
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def tabular_features(df):
    df = df.copy()
    concert_at = pd.to_datetime(df["concert_at"], utc=True)
    pred_at = pd.to_datetime(df["prediction_at"], utc=True)

    # Price
    df["total_cost"] = df["ticket_price"] * (1 - df["discount_pct"] / 100) + df["booking_fee"]
    df["cost_vs_typical"] = df["total_cost"] / df["typical_ticket_spend"]      # NaN for first-timers
    df["cost_rank_in_concert"] = df.groupby("concert_id")["ticket_price"].rank(pct=True)
    df["is_discounted"] = (df["discount_pct"] > 0).astype(int)

    # Geography
    df["distance_km"] = haversine_km(df["home_lat"], df["home_lon"], df["venue_lat"], df["venue_lon"])
    df["same_city"] = (df["home_city"] == df["venue_city"]).astype(int)

    # Timing
    df["days_to_concert"] = (concert_at - pred_at).dt.total_seconds() / 86400
    df["lead_gap"] = df["days_to_concert"] - df["usual_booking_lead_days"]    # >0: early vs habit

    # Travel search: only meaningful if it targets the venue city
    df["has_travel_search"] = df["travel_search_city"].notna().astype(int)
    df["travel_to_venue"] = (df["travel_search_city"] == df["venue_city"]).astype(int)
    arrival = pd.to_datetime(df["travel_search_arrival"], utc=True)
    df["travel_arrival_gap_days"] = np.where(df["travel_to_venue"] == 1,
                                             (concert_at - arrival).dt.total_seconds() / 86400, np.nan)
    df["travel_match_date"] = ((df["travel_to_venue"] == 1)
                               & (df["travel_arrival_gap_days"].abs() <= 3)).astype(int)

    # Engagement
    df["visits_missing"] = df["event_page_visits_7d"].isna().astype(int)

    # Stated genre vs concert genres
    df["fav_genre_missing"] = df["stated_favourite_genre"].isna().astype(int)
    df["fav_genre_in_concert"] = [int(isinstance(f, str) and f in g.split("|"))
                                  for f, g in zip(df["stated_favourite_genre"], df["genres"])]
    df["fav_genre_is_main"] = [int(isinstance(f, str) and f == g.split("|")[0])
                               for f, g in zip(df["stated_favourite_genre"], df["genres"])]
    df["n_concert_genres"] = df["genres"].str.count(r"\|") + 1

    # User-level context across the 5 offers (same user, 5 rows)
    g = df.groupby("user_id")
    df["user_total_wishlists"] = g["wishlisted"].transform("sum")
    df["user_total_visits"] = g["event_page_visits_7d"].transform("sum")
    df["visits_share_user"] = df["event_page_visits_7d"] / df["user_total_visits"].replace(0, np.nan)
    df["user_n_travel_to_venue"] = g["travel_to_venue"].transform("sum")
    df["education_ord"] = df["education"].map({"secondary": 0, "undergraduate": 1, "postgraduate": 2})
    return df


def music_features(df, ev, status, artist_genres):
    df = df.copy()
    # Account status: 4 distinct situations (unlinked / active / empty / missing)
    df["app_status"] = df["music_app_user_id"].map(status)
    df.loc[df["music_app_user_id"].isna(), "app_status"] = "unlinked"
    for s in ["unlinked", "active", "empty", "missing"]:
        df[f"app_{s}"] = (df["app_status"] == s).astype(int)

    # Overall listening
    df = df.merge(user_level_music(ev), left_on="music_app_user_id", right_index=True, how="left")

    # User x concert-artist
    ua = user_artist_music(ev)
    df = df.merge(ua, left_on=["music_app_user_id", "artist_id"],
                  right_on=["mid", "artist_id"], how="left").drop(columns="mid")
    # For an *active* account, no plays of the artist really means 0 (not unknown).
    a_cols = [c for c in ua.columns if c.startswith("a_")]
    zero_cols = [c for c in a_cols if c not in ("a_completion", "a_skip_rate", "a_search_rate",
                                                 "a_save_rate", "a_days_since_last", "a_days_since_first")]
    active = df["app_status"] == "active"
    df.loc[active, zero_cols] = df.loc[active, zero_cols].fillna(0)
    df["a_share_tracks"] = df["a_plays"] / df["u_n_tracks"]
    df["a_share_hours"] = df["a_hours"] / df["u_listen_hours"]

    # Rank of this artist among the user's 5 concert artists (1 = favourite)
    df["a_rank_in_user"] = df.groupby("user_id")["a_weighted_score"].rank(ascending=False, method="min")
    df["a_score_share_user"] = df["a_weighted_score"] / df.groupby("user_id")["a_weighted_score"].transform("sum").replace(0, np.nan)

    # User x concert genres: share of listening in the concert's genres
    gs = user_genre_shares(ev)
    gs_rows = gs.reindex(df["music_app_user_id"]).to_numpy()
    genre_idx = {g: i for i, g in enumerate(gs.columns)}
    concert_genres = df["artist_id"].map(artist_genres)
    sum_share, max_share, main_share = [], [], []
    for row, gl in zip(gs_rows, concert_genres):
        if np.isnan(row).all():
            sum_share.append(np.nan); max_share.append(np.nan); main_share.append(np.nan); continue
        vals = [row[genre_idx[g]] for g in gl if g in genre_idx]
        sum_share.append(np.sum(vals)); max_share.append(np.max(vals))
        main_share.append(row[genre_idx[gl[0]]] if gl[0] in genre_idx else 0)
    df["g_concert_genre_share"] = sum_share
    df["g_max_genre_share"] = max_share
    df["g_main_genre_share"] = main_share
    return df


# Columns that are IDs / raw strings / timestamps: not used as model inputs.
DROP = ["row_id", "user_id", "music_app_user_id", "artist_name", "genres", "concert_at",
        "sale_opened_at", "prediction_at", "travel_search_arrival", "buy", "app_status",
        "home_lat", "home_lon", "venue_lat", "venue_lon"]
CATEGORICAL = ["education", "home_city", "stated_favourite_genre", "concert_id", "artist_id",
               "venue_city", "travel_search_city"]


def build_features(data_dir="data/"):
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    ev, status, artist_genres = load_music(f"{data_dir}/music_app.json")
    out = []
    for df in (train, test):
        df = tabular_features(df)
        df = music_features(df, ev, status, artist_genres)
        out.append(df)
    return out[0], out[1]


def feature_columns(df, music=True):
    cols = [c for c in df.columns if c not in DROP]
    if not music:
        cols = [c for c in cols if not (c.startswith(("u_", "a_", "g_", "app_")))]
    return cols
