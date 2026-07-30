import pandas as pd
from dagster import (
    AssetCheckResult,
    Definitions,
    ScheduleDefinition,
    asset,
    asset_check,
    define_asset_job,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import db
import source

ACCURACY_THRESHOLD = 0.6
FEATURE_INPUT_COLUMNS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
]
RAW_TRACK_COLUMNS = [
    "track_id",
    "track_name",
    "artists",
    "track_genre",
    "popularity",
] + FEATURE_INPUT_COLUMNS


def build_track_features(raw_tracks: pd.DataFrame) -> pd.DataFrame:
    median_popularity = raw_tracks["popularity"].median()
    features = raw_tracks.copy()
    features["is_hit"] = (features["popularity"] > median_popularity).astype(int)
    return features


def train_classifier(features: pd.DataFrame) -> dict:
    x = features[FEATURE_INPUT_COLUMNS]
    labels = features["is_hit"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, test_size=0.3, random_state=42
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))

    return {"model": model, "accuracy": accuracy, "feature_columns": FEATURE_INPUT_COLUMNS}


def score_tracks(features: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    x = features[model_bundle["feature_columns"]]
    model = model_bundle["model"]
    predicted = model.predict(x)
    probability = model.predict_proba(x)[:, 1]
    return pd.DataFrame(
        {
            "track_id": features["track_id"].values,
            "track_name": features["track_name"].values,
            "predicted_label": predicted,
            "probability": probability,
            "actual_label": features["is_hit"].values,
        }
    )


@asset
def raw_tracks() -> pd.DataFrame:
    tracks = source.fetch_tracks()
    return pd.DataFrame(tracks)[RAW_TRACK_COLUMNS]


@asset
def tracks_table(raw_tracks: pd.DataFrame) -> int:
    return db.load_table(raw_tracks, "tracks")


@asset
def track_features(raw_tracks: pd.DataFrame) -> pd.DataFrame:
    return build_track_features(raw_tracks)


@asset
def track_hit_model(track_features: pd.DataFrame) -> dict:
    return train_classifier(track_features)


@asset_check(asset=track_hit_model)
def model_quality_check(track_hit_model: dict) -> AssetCheckResult:
    accuracy = track_hit_model["accuracy"]
    return AssetCheckResult(
        passed=accuracy >= ACCURACY_THRESHOLD, metadata={"accuracy": accuracy}
    )


@asset
def track_hit_predictions(track_features: pd.DataFrame, track_hit_model: dict) -> int:
    predictions = score_tracks(track_features, track_hit_model)
    return db.load_table(predictions, "track_hit_predictions")


refresh_spotify_job = define_asset_job(name="refresh_spotify_job")

refresh_spotify_weekly = ScheduleDefinition(
    name="refresh_spotify_weekly",
    job=refresh_spotify_job,
    cron_schedule="0 6 * * 1",
)

defs = Definitions(
    assets=[raw_tracks, tracks_table, track_features, track_hit_model, track_hit_predictions],
    asset_checks=[model_quality_check],
    jobs=[refresh_spotify_job],
    schedules=[refresh_spotify_weekly],
)
