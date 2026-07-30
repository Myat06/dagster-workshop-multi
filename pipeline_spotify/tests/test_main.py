from unittest.mock import patch

import pandas as pd
from dagster import materialize

import db
import source
from main import (
    raw_tracks,
    track_features,
    track_hit_model,
    track_hit_predictions,
    tracks_table,
    model_quality_check,
)


def _make_track(track_id: str, quiet: bool) -> dict:
    return {
        "track_id": track_id,
        "track_name": f"Track {track_id}",
        "artists": "Artist",
        "track_genre": "acoustic" if quiet else "pop",
        "popularity": 10 if quiet else 90,
        "danceability": 0.1 if quiet else 0.9,
        "energy": 0.1 if quiet else 0.9,
        "key": 1,
        "loudness": -25.0 if quiet else -2.0,
        "mode": 1,
        "speechiness": 0.03 if quiet else 0.15,
        "acousticness": 0.95 if quiet else 0.02,
        "instrumentalness": 0.0,
        "liveness": 0.1 if quiet else 0.4,
        "valence": 0.1 if quiet else 0.9,
        "tempo": 70.0 if quiet else 140.0,
        "duration_ms": 200000,
        "time_signature": 4,
    }


FAKE_TRACKS = [_make_track(f"q{i}", quiet=True) for i in range(10)] + [
    _make_track(f"h{i}", quiet=False) for i in range(10)
]


def test_spotify_pipeline_produces_predictions_and_passes_quality_check():
    loaded = {}

    def fake_load_table(df: pd.DataFrame, table_name: str) -> int:
        loaded[table_name] = df
        return len(df)

    with patch.object(
        source, "fetch_tracks", return_value=FAKE_TRACKS
    ), patch.object(db, "load_table", side_effect=fake_load_table):
        result = materialize(
            [
                raw_tracks,
                tracks_table,
                track_features,
                track_hit_model,
                track_hit_predictions,
                model_quality_check,
            ]
        )

    assert result.success

    assert loaded["tracks"].shape[0] == 20

    predictions = loaded["track_hit_predictions"]
    assert len(predictions) == 20
    assert set(predictions.columns) == {
        "track_id",
        "track_name",
        "predicted_label",
        "probability",
        "actual_label",
    }

    evaluations = result.get_asset_check_evaluations()
    assert len(evaluations) == 1
    assert evaluations[0].passed is True
