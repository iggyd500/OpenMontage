from pathlib import Path

from tools.video.stock_candidate_download import StockCandidateDownload


class FakeSource:
    name = "pixabay_video"
    provider = "pixabay"

    def is_available(self):
        return True

    def download(self, candidate, out_path):
        assert candidate.source_id == "91"
        Path(out_path).write_bytes(b"validated sample asset")
        return Path(out_path)


def _candidate():
    return {
        "source": "pixabay_video", "source_asset_id": "91", "source_url": "https://pixabay.com/videos/91/",
        "download_url": "https://cdn.example/91.mp4", "kind": "video",
        "resolution": {"width": 720, "height": 1280}, "duration_seconds": 13,
        "license": "Pixabay Content License", "source_tags": "doctor scan",
    }


def test_download_uses_source_adapter_and_returns_provenance(monkeypatch, tmp_path):
    projects = tmp_path / "projects"
    monkeypatch.setattr("tools.video.stock_candidate_download._projects_root", lambda: projects)
    monkeypatch.setattr("tools.video.stock_sources.get_source", lambda _name: FakeSource())
    output = projects / "demo" / "assets" / "video" / "medical.mp4"
    result = StockCandidateDownload().execute({"candidate": _candidate(), "output_path": str(output)})
    assert result.success
    assert output.read_bytes() == b"validated sample asset"
    assert result.data["provider"] == "pixabay"
    assert result.data["license"] == "Pixabay Content License"
    assert result.data["file_size_bytes"] == len(b"validated sample asset")


def test_download_refuses_overwrite_and_paths_outside_projects(monkeypatch, tmp_path):
    projects = tmp_path / "projects"
    monkeypatch.setattr("tools.video.stock_candidate_download._projects_root", lambda: projects)
    target = projects / "demo" / "keep.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"preserve")
    result = StockCandidateDownload().execute({"candidate": _candidate(), "output_path": str(target)})
    assert not result.success and "overwrite" in result.error.lower()
    assert target.read_bytes() == b"preserve"
    outside = tmp_path / "elsewhere.mp4"
    result = StockCandidateDownload().execute({"candidate": _candidate(), "output_path": str(outside)})
    assert not result.success and "projects directory" in result.error


def test_download_rejects_non_https_urls(monkeypatch, tmp_path):
    projects = tmp_path / "projects"
    monkeypatch.setattr("tools.video.stock_candidate_download._projects_root", lambda: projects)
    candidate = {**_candidate(), "download_url": "http://cdn.example/91.mp4"}
    result = StockCandidateDownload().execute({"candidate": candidate, "output_path": str(projects / "demo" / "file.mp4")})
    assert not result.success and "HTTPS" in result.error
