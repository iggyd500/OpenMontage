"""Vertical / arbitrary-resolution support for video_compose's FFmpeg compose.

Regression test for a silent-dimension bug: the compose target resolution was
resolved from `profile` (and the documented `metadata.compose_target` hook) but
the per-segment scale/pad filter hardcoded 1920x1080, so vertical profiles like
`tiktok` silently produced landscape output. These tests run the real FFmpeg
path on a tiny lavfi fixture and assert the output dimensions.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.video.video_compose import VideoCompose

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)


def _make_clip(path: Path, w: int = 1280, h: int = 720, d: int = 2) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"color=c=teal:s={w}x{h}:d={d}:r=30",
         "-c:v", "libx264", "-crf", "28", "-pix_fmt", "yuv420p",
         "-g", "30", "-keyint_min", "30", str(path)],
        capture_output=True, check=True,
    )


def _dims(path: Path) -> tuple[int, int]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)]
    ).decode().strip()
    w, h = out.split(",")
    return int(w), int(h)


def _edit_decisions(src: Path, metadata: dict | None = None) -> dict:
    ed = {
        "version": "1.0",
        "render_runtime": "ffmpeg",
        "cuts": [{"id": "c1", "source": str(src), "in_seconds": 0, "out_seconds": 2}],
    }
    if metadata:
        ed["metadata"] = metadata
    return ed


def test_compose_default_is_landscape_hd(tmp_path):
    """No profile / no target → unchanged 1920x1080 default (backward compatible)."""
    src = tmp_path / "in.mp4"
    _make_clip(src)
    out = tmp_path / "out.mp4"
    r = VideoCompose().execute(
        {"operation": "compose", "edit_decisions": _edit_decisions(src), "output_path": str(out)}
    )
    assert r.success, r.error
    assert _dims(out) == (1920, 1080)


def test_compose_vertical_profile(tmp_path):
    """profile='tiktok' → 1080x1920 (the bug: previously stayed 1920x1080)."""
    src = tmp_path / "in.mp4"
    _make_clip(src)
    out = tmp_path / "out.mp4"
    r = VideoCompose().execute(
        {"operation": "compose", "edit_decisions": _edit_decisions(src),
         "profile": "tiktok", "output_path": str(out)}
    )
    assert r.success, r.error
    assert _dims(out) == (1080, 1920)


def test_compose_target_override_cover(tmp_path):
    """metadata.compose_target with fit='cover' → exact requested dims, cropped to fill."""
    src = tmp_path / "in.mp4"
    _make_clip(src)
    out = tmp_path / "out.mp4"
    ed = _edit_decisions(src, metadata={"compose_target": {"width": 720, "height": 1280, "fit": "cover"}})
    r = VideoCompose().execute(
        {"operation": "compose", "edit_decisions": ed, "output_path": str(out)}
    )
    assert r.success, r.error
    assert _dims(out) == (720, 1280)


def test_remotion_render_accepts_low_memory_concurrency(tmp_path, monkeypatch):
    """The registered Remotion path can be constrained for low-memory hosts."""
    tool = VideoCompose()
    captured = {}
    monkeypatch.setattr(tool, "_stage_remotion_media", lambda props, public_dir: 0)

    def fake_run_command(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[5]).write_bytes(b"rendered")

    monkeypatch.setattr(tool, "run_command", fake_run_command)
    result = tool.execute(
        {
            "operation": "remotion_render",
            "composition_data": {
                "renderer_family": "explainer-data",
                "cuts": [{"id": "c1", "source": "missing.mp4", "in_seconds": 0, "out_seconds": 1}],
            },
            "output_path": str(tmp_path / "low-memory.mp4"),
            "remotion_concurrency": 1,
        }
    )

    assert result.success, result.error
    assert "--concurrency=1" in captured["cmd"]


def test_remotion_render_rejects_invalid_concurrency(tmp_path, monkeypatch):
    """Invalid concurrency values fail before starting the renderer."""
    tool = VideoCompose()
    monkeypatch.setattr(tool, "_stage_remotion_media", lambda props, public_dir: 0)
    monkeypatch.setattr(tool, "run_command", lambda *args, **kwargs: pytest.fail("renderer should not start"))
    result = tool.execute(
        {
            "operation": "remotion_render",
            "composition_data": {"renderer_family": "explainer-data", "cuts": []},
            "output_path": str(tmp_path / "invalid.mp4"),
            "remotion_concurrency": 0,
        }
    )

    assert not result.success
    assert "positive integer" in result.error
