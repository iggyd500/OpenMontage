"""Mocked contract tests for ElevenLabs forced alignment."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis.elevenlabs_forced_alignment import ElevenLabsForcedAlignment  # noqa: E402


def _payload():
    return {
        "words": [
            {"text": "Hello", "start": 0.1, "end": 0.45, "loss": 0.01},
            {"text": " ", "start": 0.45, "end": 0.5, "loss": 0.02},
            {"text": "world!", "start": 0.5, "end": 0.9, "loss": 0.02},
        ],
        "characters": [
            {"text": "H", "start": 0.1, "end": 0.2},
            {"text": "i", "start": 0.2, "end": 0.3},
        ],
        "loss": 0.015,
    }


def test_alignment_request_and_sidecar(monkeypatch, tmp_path):
    secret = "unit-test-secret"
    monkeypatch.setenv("ELEVENLABS_API_KEY", secret)
    monkeypatch.setattr(
        "tools.analysis.elevenlabs_forced_alignment.probe_duration", lambda _: 1.0
    )
    response = Mock(json=Mock(return_value=_payload()))
    response.raise_for_status.return_value = None
    post = Mock(return_value=response)
    monkeypatch.setattr("tools.analysis.elevenlabs_forced_alignment.requests.post", post)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    output = tmp_path / "alignment.json"

    result = ElevenLabsForcedAlignment().execute(
        {"audio_path": str(audio), "text": "Hello world!", "output_path": str(output)}
    )

    assert result.success, result.error
    assert post.call_args.args[0] == "https://api.elevenlabs.io/v1/forced-alignment"
    assert post.call_args.kwargs["headers"] == {"xi-api-key": secret}
    assert post.call_args.kwargs["data"] == {"text": "Hello world!"}
    assert [w["word"] for w in result.data["word_timestamps"]] == ["Hello", "world!"]
    assert result.data["audio_duration_seconds"] == 1.0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["alignment_loss"] == 0.015
    assert saved["raw_api_response"] == _payload()
    quality_path = tmp_path / "alignment.quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality["overall_alignment_loss"] == 0.015
    assert quality["maximum_alignment_loss"] == 0.02
    assert quality["raw_api_response_preserved"] is True
    assert quality["subtitles_fully_validated"] is False


def test_alignment_rejects_missing_words_and_audio_boundary(monkeypatch, tmp_path):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.setattr(
        "tools.analysis.elevenlabs_forced_alignment.probe_duration", lambda _: 1.0
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    for index, words in enumerate(([], [{"text": "late", "start": 0.8, "end": 1.1}])):
        payload = {**_payload(), "words": words}
        response = Mock(json=Mock(return_value=payload))
        response.raise_for_status.return_value = None
        monkeypatch.setattr(
            "tools.analysis.elevenlabs_forced_alignment.requests.post",
            Mock(return_value=response),
        )
        result = ElevenLabsForcedAlignment().execute(
            {"audio_path": str(audio), "text": "Hello world!", "output_path": str(tmp_path / f"out-{index}.json")}
        )
        assert not result.success
        assert "alignment" in result.error.lower() or "duration" in result.error.lower()


def test_alignment_rejects_missing_or_duplicated_script_words(monkeypatch, tmp_path):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.setattr(
        "tools.analysis.elevenlabs_forced_alignment.probe_duration", lambda _: 1.0
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    response = Mock(json=Mock(return_value=_payload()))
    response.raise_for_status.return_value = None
    monkeypatch.setattr(
        "tools.analysis.elevenlabs_forced_alignment.requests.post", Mock(return_value=response)
    )
    result = ElevenLabsForcedAlignment().execute({
        "audio_path": str(audio), "text": "Hello world again!",
        "output_path": str(tmp_path / "out.json"),
    })
    assert not result.success
    assert "does not match" in result.error


def test_quality_report_flags_missing_words_and_keeps_duplicate_occurrences():
    report = ElevenLabsForcedAlignment.build_quality_report(
        "go go now", [
            {"word": "go", "start": 0.0, "end": 0.2, "loss": 0.1},
            {"word": "now", "start": 0.3, "end": 0.5, "loss": 0.2},
        ],
        0.15,
        1.0,
    )
    assert report["total_transcript_words"] == 3
    assert report["successfully_aligned_words"] == 2
    assert report["missing_words"] == ["go"]
    assert report["unmatched_words"] == []
    assert report["quality_status"] == "blocked"
    assert report["subtitles_fully_validated"] is False


def test_quality_report_ranks_high_loss_without_claiming_a_threshold():
    report = ElevenLabsForcedAlignment.build_quality_report(
        "alpha beta gamma", [
            {"word": "alpha", "start": 0.0, "end": 0.2, "loss": 0.2},
            {"word": "beta", "start": 0.3, "end": 0.5, "loss": 1.4},
            {"word": "gamma", "start": 0.6, "end": 0.9, "loss": 0.8},
        ],
        0.8,
        1.0,
    )
    candidates = report["highest_loss_review_candidates"]
    assert [word["word"] for word in candidates] == ["beta", "gamma", "alpha"]
    assert report["maximum_alignment_loss"] == 1.4
    assert "relative ranking only" in report["loss_review_policy"]
    assert report["quality_status"] == "human_review_required"


def test_quality_report_flags_invalid_timestamps_and_duration_candidates():
    report = ElevenLabsForcedAlignment.build_quality_report(
        "one two three four five", [
            {"word": "one", "start": -0.1, "end": 0.1, "loss": 0.1},
            {"word": "two", "start": 0.2, "end": 0.25, "loss": 0.2},
            {"word": "three", "start": 0.3, "end": 0.5, "loss": 0.3},
            {"word": "four", "start": 0.6, "end": 0.9, "loss": 0.4},
            {"word": "five", "start": 1.0, "end": 2.2, "loss": 0.5},
        ],
        0.3,
        2.0,
    )
    assert report["quality_status"] == "blocked"
    assert report["invalid_timestamp_words"][0]["word"] == "one"
    assert report["duration_review"]["short_duration_candidates"]
    assert report["duration_review"]["long_duration_candidates"]
    assert report["duration_review"]["thresholds_are_provider_defined"] is False


def test_quality_report_handles_empty_or_unmatched_alignment():
    report = ElevenLabsForcedAlignment.build_quality_report(
        "expected words", [], None, 1.0
    )
    assert report["missing_words"] == ["expected", "words"]
    assert report["unmatched_words"] == []
    assert report["maximum_alignment_loss"] is None
    assert report["quality_status"] == "blocked"


def test_alignment_error_detail_redacts_credential(monkeypatch, tmp_path):
    secret = "private-test-token"
    monkeypatch.setenv("ELEVENLABS_API_KEY", secret)
    monkeypatch.setattr(
        "tools.analysis.elevenlabs_forced_alignment.probe_duration", lambda _: 1.0
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    response = Mock()
    response.json.return_value = {"detail": {"message": f"Rejected {secret}"}}
    response.status_code = 401
    response.raise_for_status.side_effect = requests.HTTPError("bad", response=response)
    monkeypatch.setattr(
        "tools.analysis.elevenlabs_forced_alignment.requests.post", Mock(return_value=response)
    )

    result = ElevenLabsForcedAlignment().execute(
        {"audio_path": str(audio), "text": "Hello", "output_path": str(tmp_path / "out.json")}
    )
    assert not result.success
    assert secret not in result.error
    assert "HTTP 401" in result.error


def test_alignment_requires_credentials_and_never_overwrites(monkeypatch, tmp_path):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    tool = ElevenLabsForcedAlignment()
    result = tool.execute({"audio_path": "missing.wav", "text": "Hi", "output_path": "out.json"})
    assert not result.success and "not configured" in result.error

    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    output = tmp_path / "alignment.json"
    output.write_text("keep", encoding="utf-8")
    result = tool.execute({"audio_path": str(audio), "text": "Hi", "output_path": str(output)})
    assert not result.success
    assert output.read_text(encoding="utf-8") == "keep"


def test_cost_is_quote_required_not_reported_as_free():
    quote = ElevenLabsForcedAlignment().dry_run(
        {"audio_path": "audio.wav", "text": "text", "output_path": "out.json"}
    )
    assert quote["cost_status"] == "quote_required"


def test_registry_discovers_forced_alignment_tool():
    from tools.tool_registry import ToolRegistry

    registry = ToolRegistry()
    discovered = registry.discover()
    assert "elevenlabs_forced_alignment" in discovered
    assert registry.get("elevenlabs_forced_alignment").capability == "speech_alignment"
