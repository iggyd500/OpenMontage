from types import SimpleNamespace

import pytest

from tools.video.stock_candidate_search import StockCandidateSearch, _candidate_metadata
from tools.video.stock_sources import Candidate


class FakeSource:
    name = "pixabay_video"
    provider = "pixabay"

    def is_available(self):
        return True

    def search(self, query, filters):
        self.seen = (query, filters)
        return [Candidate(
            source=self.name, source_id="42", source_url="https://pixabay.com/videos/42/",
            download_url="https://cdn.example/42.mp4", kind="video", width=720,
            height=1280, duration=18, creator="uploader", license="Pixabay Content License",
            source_tags="doctor, medical scan", thumbnail_url="https://cdn.example/42.jpg",
            extra={"views": 20},
        )]


def test_search_returns_provider_metadata_without_downloading(monkeypatch):
    source = FakeSource()
    monkeypatch.setattr("tools.video.stock_sources.get_source", lambda _name: source)
    result = StockCandidateSearch().execute({
        "source": "pixabay_video", "query": "doctor examining medical scans",
        "kind": "video", "per_page": 6, "orientation": "portrait",
    })
    assert result.success
    assert result.data["candidate_count"] == 1
    assert result.data["downloaded"] is False
    candidate = result.data["candidates"][0]
    assert candidate["asset_id"] == "pixabay_video_42"
    assert candidate["description"] is None
    assert candidate["tags"] == ["doctor", "medical", "scan"]
    assert candidate["duration_seconds"] == 18
    assert candidate["resolution"] == {"width": 720, "height": 1280}
    assert candidate["orientation"] == "portrait"
    assert candidate["download_url"].startswith("https://")
    assert candidate["source_url"].startswith("https://")
    assert candidate["license"] == "Pixabay Content License"
    assert source.seen[1].per_page == 6


@pytest.mark.parametrize("status,category", [(429, "rate_limited"), (500, "provider_error")])
def test_provider_errors_are_structured_and_redacted(monkeypatch, status, category):
    secret = "not-for-output"
    response = SimpleNamespace(status_code=status)

    class FailingSource(FakeSource):
        def search(self, *_args):
            raise pytest.importorskip("requests").HTTPError(f"failed https://host/?key={secret}", response=response)

    monkeypatch.setattr("tools.video.stock_sources.get_source", lambda _name: FailingSource())
    result = StockCandidateSearch().execute({"source": "pixabay_video", "query": "medical scans"})
    assert not result.success
    assert result.data["error_category"] == category
    assert result.data["http_status"] == status
    assert secret not in result.error
    assert "key=" not in result.error


def test_unknown_or_unavailable_source_does_not_fallback(monkeypatch):
    monkeypatch.setattr("tools.video.stock_sources.get_source", lambda _name: (_ for _ in ()).throw(KeyError("x")))
    result = StockCandidateSearch().execute({"source": "missing", "query": "specific visible action"})
    assert not result.success and "Unknown stock source" in result.error


def test_candidate_metadata_does_not_invent_license_or_description():
    source = FakeSource()
    candidate = Candidate(
        source=source.name, source_id="43", source_url="", download_url="",
        kind="image", extra={},
    )
    metadata = _candidate_metadata(source, candidate, "query")
    assert metadata["license"] is None
    assert metadata["description"] is None
    assert metadata["resolution"] is None
    assert metadata["orientation"] is None
