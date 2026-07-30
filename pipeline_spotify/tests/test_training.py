import pandas as pd

from main import ACCURACY_THRESHOLD, FEATURE_INPUT_COLUMNS, model_quality_check, train_classifier

# Clearly separable on acousticness/danceability so the classifier's accuracy
# is deterministic regardless of the train/test split.
SEPARABLE_FEATURES = pd.DataFrame(
    {
        "track_id": [f"t{i}" for i in range(1, 11)],
        "track_name": [f"Track {i}" for i in range(1, 11)],
        "danceability": [0.1] * 5 + [0.9] * 5,
        "energy": [0.1] * 5 + [0.9] * 5,
        "key": [1] * 10,
        "loudness": [-25.0] * 5 + [-2.0] * 5,
        "mode": [1] * 10,
        "speechiness": [0.03] * 5 + [0.15] * 5,
        "acousticness": [0.95] * 5 + [0.02] * 5,
        "instrumentalness": [0.0] * 10,
        "liveness": [0.1] * 5 + [0.4] * 5,
        "valence": [0.1] * 5 + [0.9] * 5,
        "tempo": [70.0] * 5 + [140.0] * 5,
        "duration_ms": [200000] * 10,
        "time_signature": [4] * 10,
        "is_hit": [0] * 5 + [1] * 5,
    }
)


def test_train_classifier_meets_accuracy_threshold_on_separable_data():
    bundle = train_classifier(SEPARABLE_FEATURES)

    assert bundle["accuracy"] >= ACCURACY_THRESHOLD
    assert bundle["feature_columns"] == FEATURE_INPUT_COLUMNS
    assert hasattr(bundle["model"], "predict")


def test_model_quality_check_fails_below_accuracy_threshold():
    low_accuracy_bundle = {"model": None, "accuracy": 0.3, "feature_columns": []}

    result = model_quality_check(low_accuracy_bundle)

    assert result.passed is False
