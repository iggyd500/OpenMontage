import types

from tools.video.stock_candidate_preview import MAX_PREVIEW_BYTES, StockCandidatePreview


class Response:
    headers = {"Content-Type": "image/jpeg", "Content-Length": "4"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield b"jpeg"


def test_preview_download_is_small_and_project_scoped(monkeypatch, tmp_path):
    projects = tmp_path / "projects"
    monkeypatch.setattr("tools.video.stock_candidate_preview._projects_root", lambda: projects)
    monkeypatch.setitem(__import__("sys").modules, "requests", types.SimpleNamespace(get=lambda *_a, **_k: Response()))
    output = projects / "demo" / "artifacts" / "preview.jpg"
    result = StockCandidatePreview().execute({
        "candidate": {"asset_id": "pixabay_video_1", "thumbnail_url": "https://cdn.example/thumb.jpg"},
        "output_path": str(output),
    })
    assert result.success
    assert output.read_bytes() == b"jpeg"
    assert result.data["file_size_bytes"] == 4
    assert result.data["max_preview_bytes"] == MAX_PREVIEW_BYTES


def test_preview_rejects_non_image_response_and_overwrite(monkeypatch, tmp_path):
    projects = tmp_path / "projects"
    monkeypatch.setattr("tools.video.stock_candidate_preview._projects_root", lambda: projects)
    bad = Response()
    bad.headers = {"Content-Type": "video/mp4"}
    monkeypatch.setitem(__import__("sys").modules, "requests", types.SimpleNamespace(get=lambda *_a, **_k: bad))
    result = StockCandidatePreview().execute({
        "candidate": {"thumbnail_url": "https://cdn.example/not-image"},
        "output_path": str(projects / "demo" / "preview.jpg"),
    })
    assert not result.success and "did not return an image" in result.error
    result_path = projects / "demo" / "keep.jpg"
    result_path.parent.mkdir(parents=True)
    result_path.write_bytes(b"existing")
    result = StockCandidatePreview().execute({
        "candidate": {"thumbnail_url": "https://cdn.example/thumb.jpg"},
        "output_path": str(result_path),
    })
    assert not result.success and "overwrite" in result.error.lower()


def test_wikimedia_preview_uses_source_adapter_user_agent(monkeypatch, tmp_path):
    projects = tmp_path / "projects"
    monkeypatch.setattr("tools.video.stock_candidate_preview._projects_root", lambda: projects)
    seen = {}

    def get(_url, **kwargs):
        seen.update(kwargs)
        return Response()

    monkeypatch.setitem(__import__("sys").modules, "requests", types.SimpleNamespace(get=get))
    result = StockCandidatePreview().execute({
        "candidate": {"source": "wikimedia", "thumbnail_url": "https://upload.wikimedia.org/thumb.jpg"},
        "output_path": str(projects / "demo" / "commons.jpg"),
    })
    assert result.success
    assert "OpenMontageBot" in seen["headers"]["User-Agent"]
