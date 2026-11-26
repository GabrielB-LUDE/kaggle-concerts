import numpy as np


def add_ranking_features(df):
    """Compare the five concert artists offered to each user."""
    df = df.copy()

    # Group the five concert opportunities of each user.
    g = df.groupby("user_id")

    # Rank the artists based on the number of plays.
    # ascending=False means the most played artist is rank 1.
    # method="min" gives tied artists the same best rank.
    df["artist_rank"] = g["artist_plays"].rank(ascending=False, method="min")

    # Total plays across the five concert artists.
    # transform("sum") keeps the original number of rows.
    df["total_concert_artist_plays"] = g["artist_plays"].transform("sum", min_count=1)

    # The artist's share among the five concert artists.
    df["artist_relative_preference"] = (
        df["artist_plays"] / df["total_concert_artist_plays"].replace(0, np.nan))

    # No listening to any concert artist means no meaningful rank.
    df.loc[df["total_concert_artist_plays"] == 0, "artist_rank"] = np.nan

    # Rank artists based on listening during the last 30 days.
    df["artist_recent_rank"] = g["artist_plays_30d"].rank(ascending=False, method="min")

    # Remove meaningless recent ranks when all counts are zero.
    recent_total = g["artist_plays_30d"].transform("sum", min_count=1)
    df.loc[recent_total == 0, "artist_recent_rank"] = np.nan

    return df


def add_behaviour_features(df):
    """Purchase intention, price sensitivity, purchase history,
    missing signals and comparison between the five concerts."""
    df = df.copy()

    # ==============================
    # 1. PURCHASE INTENTION
    # ==============================

    # Retrieve visits while preserving originally missing values.
    visits = df["event_page_visits_7d"].mask(df["Event_pages_missing"] == 1)

    # Combination of wishlist and browsing activity.
    df["wishlist_and_visits"] = df["wishlisted"] * visits
    # Combination of wishlist and discount.
    df["wishlist_and_discount"] = df["wishlisted"] * df["discount_pct"]
    # Number of visits without adding the concert to the wishlist.
    df["visits_without_wishlist"] = visits * (1 - df["wishlisted"])

    # ==============================
    # 2. PRICE SENSITIVITY
    # ==============================

    # Actual amount of money saved.
    df["discount_amount"] = df["ticket_price"] * df["discount_pct"] / 100
    # Percentage of the total cost represented by booking fees.
    df["booking_fee_ratio"] = df["booking_fee"] / df["total_cost"].replace(0, np.nan)

    # Retrieve usual spending while preserving missing information.
    usual_spend = df["typical_ticket_spend"].mask(df["spend_missing"] == 1)
    # How much more or less expensive the concert is compared with usual spending.
    df["price_spend_gap"] = df["total_cost"] - usual_spend

    # Does musical attachment coexist with an expensive offer?
    df["artist_affinity_price"] = df["artist_play_pct"] * df["cost_vs_typical"]

    # ==============================
    # 3. PURCHASE HISTORY
    # ==============================

    # Reduce the influence of extremely frequent buyers.
    df["purchase_count_log"] = np.log1p(df["past_purchase_count"])
    # Are experienced buyers sensitive to discounts?
    df["loyal_buyer_discount"] = df["purchase_count_log"] * df["discount_pct"]
    # Is an experienced buyer showing explicit interest?
    df["loyal_buyer_wishlist"] = df["purchase_count_log"] * df["wishlisted"]
    # Does a discount encourage a first-time customer?
    df["first_timer_discount"] = df["first_timer"] * df["discount_pct"]

    # ==============================
    # 4. SILENCE AND CONTRADICTIONS
    # ==============================

    # Unknown browsing activity, but concert wishlisted.
    df["silent_but_interested"] = ((df["Event_pages_missing"] == 1) & (df["wishlisted"] == 1)).astype(int)
    # We know that the customer did not visit the page.
    df["zero_visits"] = (visits == 0).astype(int)
    # Musical affinity without explicit wishlist interest.
    df["fan_without_wishlist"] = df["artist_play_pct"] * (1 - df["wishlisted"])
    # Music account exists, but listening history is unavailable.
    df["music_history_missing"] = (df["app_status"] == "missing").astype(int)
    # Music account exists, but has no recorded listening activity.
    df["music_account_empty"] = (df["app_status"] == "empty").astype(int)
    # Purchase history is unavailable.
    df["purchase_history_missing"] = df["past_purchase_count"].isna().astype(int)

    # Count selected missing-information signals.
    missing_columns = ["spend_missing", "Event_pages_missing", "travel_search_date_unknown",
                       "music_history_missing", "purchase_history_missing"]
    df["missing_information_count"] = df[missing_columns].sum(axis=1)

    # ==============================
    # 5. COMPARE THE FIVE CONCERTS
    # ==============================

    # Group the five concert offers for each customer.
    g = df.groupby("user_id")

    # Rank concerts from cheapest to most expensive.
    df["price_rank"] = g["total_cost"].rank(ascending=True, method="min")
    # Rank concerts from highest to lowest discount.
    df["discount_rank"] = g["discount_pct"].rank(ascending=False, method="min")

    # Count how many concerts the customer wishlisted.
    df["total_wishlisted"] = g["wishlisted"].transform("sum", min_count=1)
    # Is this the only concert wishlisted?
    df["only_wishlisted_concert"] = ((df["total_wishlisted"] == 1) & (df["wishlisted"] == 1)).astype(int)

    # Share of observed browsing dedicated to this concert.
    total_visits = visits.groupby(df["user_id"]).transform("sum", min_count=1)
    df["visits_share"] = visits / total_visits.replace(0, np.nan)
    # Rank concerts by browsing interest.
    df["visits_rank"] = visits.groupby(df["user_id"]).rank(ascending=False, method="min")

    # Difference in price compared with the cheapest concert.
    df["price_above_cheapest"] = df["total_cost"] - g["total_cost"].transform("min")

    return df
