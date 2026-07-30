import pandas as pd
import pytest

import source


def test_fetch_tracks_returns_parsed_records():
    fake_df = pd.DataFrame([{"track_id": "abc", "popularity": 50}])

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(source.pd, "read_csv", lambda path: fake_df)
        result = source.fetch_tracks()

    assert result == [{"track_id": "abc", "popularity": 50}]


def test_fetch_tracks_raises_source_unavailable_on_read_error():
    with pytest.MonkeyPatch.context() as monkeypatch:
        def raise_os_error(path):
            raise OSError("boom")

        monkeypatch.setattr(source.pd, "read_csv", raise_os_error)

        with pytest.raises(source.SourceUnavailableError):
            source.fetch_tracks()
