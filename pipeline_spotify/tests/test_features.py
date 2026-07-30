import pandas as pd

from main import build_track_features

RAW_TRACKS = pd.DataFrame(
    [
        {
            "track_id": "t1",
            "track_name": "Quiet Song",
            "artists": "Artist A",
            "track_genre": "acoustic",
            "popularity": 10,
            "danceability": 0.3,
            "energy": 0.2,
            "key": 1,
            "loudness": -20.0,
            "mode": 1,
            "speechiness": 0.03,
            "acousticness": 0.9,
            "instrumentalness": 0.0,
            "liveness": 0.1,
            "valence": 0.2,
            "tempo": 80.0,
            "duration_ms": 200000,
            "time_signature": 4,
        },
        {
            "track_id": "t2",
            "track_name": "Loud Banger",
            "artists": "Artist B",
            "track_genre": "pop",
            "popularity": 90,
            "danceability": 0.8,
            "energy": 0.9,
            "key": 5,
            "loudness": -3.0,
            "mode": 0,
            "speechiness": 0.1,
            "acousticness": 0.05,
            "instrumentalness": 0.0,
            "liveness": 0.3,
            "valence": 0.8,
            "tempo": 128.0,
            "duration_ms": 210000,
            "time_signature": 4,
        },
    ]
)


def test_build_track_features_labels_above_median_as_hit():
    result = build_track_features(RAW_TRACKS)

    assert "is_hit" in result.columns
    # median(popularity) of [10, 90] is 50, so only the higher-popularity
    # track (t2) is above the median and labeled a hit.
    row_t1 = result.loc[result["track_id"] == "t1"].iloc[0]
    assert row_t1["is_hit"] == 0
    row_t2 = result.loc[result["track_id"] == "t2"].iloc[0]
    assert row_t2["is_hit"] == 1
