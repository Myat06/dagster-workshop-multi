from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "tracks.csv"


class SourceUnavailableError(Exception):
    """Raised when the bundled tracks dataset cannot be read."""


def fetch_tracks() -> list[dict]:
    try:
        df = pd.read_csv(DATA_PATH)
    except OSError as exc:
        raise SourceUnavailableError(
            f"Could not read bundled dataset at {DATA_PATH}"
        ) from exc
    return df.to_dict(orient="records")
