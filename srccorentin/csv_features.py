import numpy as np
import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    """Geographical distance (in km) between two points
    using their latitude and longitude."""

    # Convert geographical coordinates into radians.
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Apply the Haversine mathematical formula.
    a = (np.sin((lat2 - lat1) / 2)**2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2)

    # 6371 is the approximate radius of Earth in km.
    return 6371 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))


def add_csv_features(df):
    """Add date, price, geographical and missing-value features."""

    # Protect the original DataFrame.
    df = df.copy()

    # ----------------------------------------
    # 1. DATE FEATURES
    # ----------------------------------------

    # Convert each date column from text to datetime.
    for column in ["sale_opened_at", "concert_at", "prediction_at", "travel_search_arrival"]:
        df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)

    # Calculate the number of days until the concert.
    df["days_to_concert"] = (df["concert_at"] - df["prediction_at"]).dt.total_seconds() / 86400

    # Compare the time until the concert with the user's usual booking habits.
    df["lead_gap"] = df["days_to_concert"] - df["usual_booking_lead_days"]

    # ----------------------------------------
    # 2. PRICE FEATURES
    # ----------------------------------------

    # Calculate the actual ticket cost, including discounts and booking fees.
    df["total_cost"] = df["ticket_price"] * (1 - df["discount_pct"] / 100) + df["booking_fee"]

    # Compare the ticket cost with what the user usually spends.
    # Replace zero with NaN to avoid dividing by zero.
    df["cost_vs_typical"] = df["total_cost"] / df["typical_ticket_spend"].replace(0, np.nan)

    # Identify whether the ticket is discounted. 1 = discount, 0 = no discount.
    df["is_discounted"] = (df["discount_pct"] > 0).astype(int)

    # ----------------------------------------
    # 3. GEOGRAPHICAL FEATURES
    # ----------------------------------------

    # Distance between the user's home and the concert venue.
    df["distance_km"] = haversine(df["home_lat"], df["home_lon"], df["venue_lat"], df["venue_lon"])

    # Does the user live in the same city as the concert?
    df["same_city"] = (df["home_city"] == df["venue_city"]).astype(int)

    # Did the user search for travel to the concert city?
    df["travel_to_venue"] = (df["travel_search_city"] == df["venue_city"]).astype(int)

    # ----------------------------------------
    # 4. MISSING VALUE INDICATORS
    # ----------------------------------------

    # 1 = information missing, 0 = available.
    df["spend_missing"] = df["typical_ticket_spend"].isna().astype(int)
    df["Event_pages_missing"] = df["event_page_visits_7d"].isna().astype(int)
    df["Has_music_app"] = df["music_app_user_id"].notna().astype(int)
    df["travel_search_date_unknown"] = df["travel_search_arrival"].isna().astype(int)

    return df
