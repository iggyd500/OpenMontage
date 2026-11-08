from __future__ import annotations

import base64
import io
import json
import wave
from unittest.mock import Mock

import pytest

from tools.audio.elevenlabs_tts import ElevenLabsTTS
from tools.audio.google_tts import GoogleTTS
from tools.audio.piper_tts import PiperTTS
from tools.audio.tts_selector import TTSSelector
from tools.base_tool import ToolResult, ToolStatus


def _wav_payload(*, sample_rate: int = 44100, frames: int = 100) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(b"\0\0" * frames)
    return buffer.getvalue()


def test_configured_tts_selector_routes_to_brian_direct_without_fallback(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "unit-test-secret")
    direct = ElevenLabsTTS()
    piper = PiperTTS()
    selector = TTSSelector()
    monkeypatch.setattr(selector, "_providers", lambda: [direct, piper])
    monkeypatch.setattr(direct, "get_status", lambda: ToolStatus.AVAILABLE)
    monkeypatch.setattr(piper, "get_status", lambda: ToolStatus.AVAILABLE)
    direct_call = Mock(return_value=ToolResult(success=True, data={}))
    piper_call = Mock(return_value=ToolResult(success=True, data={}))
    monkeypatch.setattr(direct, "execute", direct_call)
    monkeypatch.setattr(piper, "execute", piper_call)

    result = selector.execute({"text": "A short narration."})

    assert result.success
    assert result.data["selected_tool"] == "elevenlabs_tts"
    assert result.data["selected_provider"] == "elevenlabs"
    call = direct_call.call_args.args[0]
    assert call["model_id"] == "eleven_multilingual_v2"
    assert call["voice_id"] == "nPczCjzI2devNBz1zQrb"
    assert call["voice_name"] == "Brian"
    assert call["language"] == "en"
    assert call["output_format"] == "wav_44100"
    piper_call.assert_not_called()


def test_selector_forwards_timestamp_request_to_configured_direct_tool(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "unit-test-secret")
    direct = ElevenLabsTTS()
    selector = TTSSelector()
    monkeypatch.setattr(selector, "_providers", lambda: [direct])
    monkeypatch.setattr(direct, "get_status", lambda: ToolStatus.AVAILABLE)
    direct_call = Mock(return_value=ToolResult(success=True, data={"word_timestamps": []}))
    monkeypatch.setattr(direct, "execute", direct_call)

    result = selector.execute({"text": "A short narration.", "timestamps": True})

    assert result.success
    assert direct_call.call_args.args[0]["with_timestamps"] is True
    assert result.data["selected_tool"] == "elevenlabs_tts"


def test_configured_provider_unavailability_does_not_fall_back_to_piper(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "unit-test-secret")
    direct = ElevenLabsTTS()
    piper = PiperTTS()
    selector = TTSSelector()
    monkeypatch.setattr(selector, "_providers", lambda: [direct, piper])
    monkeypatch.setattr(direct, "get_status", lambda: ToolStatus.UNAVAILABLE)
    monkeypatch.setattr(piper, "get_status", lambda: ToolStatus.AVAILABLE)
    direct_call = Mock()
    piper_call = Mock()
    monkeypatch.setattr(direct, "execute", direct_call)
    monkeypatch.setattr(piper, "execute", piper_call)

    result = selector.execute({"text": "A short narration."})

    assert not result.success
    assert "Configured TTS provider 'elevenlabs' is unavailable" in result.error
    direct_call.assert_not_called()
    piper_call.assert_not_called()


def test_wav_44100_uses_standard_mp3_response_and_local_ffmpeg_conversion(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "unit-test-secret")
    response = Mock(content=b"mock-mp3", headers={"request-id": "request-441"})
    response.raise_for_status = Mock()
    post = Mock(return_value=response)
    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr("shutil.which", Mock(return_value="ffmpeg"))

    def fake_ffmpeg(command, **_kwargs):
        Path(command[-1]).write_bytes(_wav_payload(sample_rate=44100))
        return Mock(returncode=0)

    from pathlib import Path

    ffmpeg = Mock(side_effect=fake_ffmpeg)
    monkeypatch.setattr("subprocess.run", ffmpeg)
    output = tmp_path / "brian.wav"

    result = ElevenLabsTTS().execute(
        {
            "text": "Hello there.",
            "voice_id": "nPczCjzI2devNBz1zQrb",
            "model_id": "eleven_multilingual_v2",
            "output_format": "wav_44100",
            "output_path": str(output),
        }
    )

    assert result.success, result.error
    assert post.call_args.kwargs["params"]["output_format"] == "mp3_44100_128"
    assert "-ar" in ffmpeg.call_args.args[0]
    assert ffmpeg.call_args.args[0][ffmpeg.call_args.args[0].index("-ar") + 1] == "44100"
    assert result.data["format"] == "wav_44100"
    assert result.data["api_output_format"] == "mp3_44100_128"
    assert result.data["sample_rate"] == 44100
    assert result.data["voice_name"] == "Brian"
    assert result.data["language"] == "en"
    with wave.open(str(output), "rb") as wav:
        assert wav.getframerate() == 44100
        assert wav.getnchannels() == 1


def test_wav_44100_missing_ffmpeg_fails_before_paid_request(monkeypatch, tmp_path):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "unit-test-secret")
    monkeypatch.setattr("shutil.which", Mock(return_value=None))
    post = Mock()
    monkeypatch.setattr("requests.post", post)

    result = ElevenLabsTTS().execute(
        {"text": "Hello", "output_format": "wav_44100", "output_path": str(tmp_path / "x.wav")}
    )

    assert not result.success
    assert "FFmpeg is required" in result.error
    post.assert_not_called()


@pytest.mark.parametrize(
    "model,limit",
    [
        ("eleven_v4", 2_000),
        ("eleven_v4_turbo", 2_000),
        ("eleven_v3", 5_000),
        ("eleven_multilingual_v2", 10_000),
        ("eleven_turbo_v2_5", 40_000),
        ("eleven_flash_v2_5", 40_000),
    ],
)
def test_model_character_limits_are_enforced_before_request(
    monkeypatch, model, limit
):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "unit-test-secret")
    post = Mock()
    monkeypatch.setattr("requests.post", post)
    result = ElevenLabsTTS().execute({"text": "x" * (limit + 1), "model_id": model})
    assert not result.success
    assert str(limit) in result.error
    post.assert_not_called()


def test_model_cost_estimates_are_model_aware_and_marked(monkeypatch, tmp_path):
    tool = ElevenLabsTTS()
    assert tool.estimate_cost({"text": "x" * 1000, "model_id": "eleven_multilingual_v2"}) == 0.08
    assert tool.estimate_cost({"text": "x" * 1000, "model_id": "eleven_flash_v2_5"}) == 0.04
    monkeypatch.setenv("ELEVENLABS_API_KEY", "unit-test-secret")
    monkeypatch.setattr(
        "requests.post", Mock(return_value=Mock(content=b"audio", headers={}))
    )
    result = tool.execute(
        {"text": "hello", "model_id": "eleven_flash_v2_5", "output_path": str(tmp_path / "x.mp3")}
    )
    assert result.success
    assert result.data["cost_status"] == "estimated"
    assert result.data["cost_is_estimate"] is True
    assert "Estimate only" in result.data["cost_estimate_note"]


def _alignment(text: str) -> dict:
    chars = list(text)
    return {
        "characters": chars,
        "character_start_times_seconds": [i / 10 for i in range(len(chars))],
        "character_end_times_seconds": [(i + 1) / 10 for i in range(len(chars))],
    }


def test_selector_routes_timestamps_stitching_and_alignment(monkeypatch, tmp_path):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "unit-test-secret")
    text = "One consistent voice."
    body = {
        "audio_base64": base64.b64encode(b"fake-mp3").decode(),
        "alignment": _alignment(text),
    }
    post = Mock(return_value=Mock(json=Mock(return_value=body), headers={"request-id": "req-01"}))
    monkeypatch.setattr("requests.post", post)
    direct = ElevenLabsTTS()
    selector = TTSSelector()
    monkeypatch.setattr(selector, "_providers", lambda: [direct])
    monkeypatch.setattr(selector, "_select_best_tool", lambda *_: (direct, None))
    out = tmp_path / "narration.mp3"
    result = selector.execute(
        {
            "text": text,
            "preferred_tool": "elevenlabs_tts",
            "model_id": "eleven_multilingual_v2",
            "voice_id": "stable-voice",
            "stability": 0.8,
            "similarity_boost": 0.9,
            "style": 0.1,
            "speed": 1.0,
            "use_speaker_boost": True,
            "previous_text": "Earlier narration.",
            "next_text": "Following narration.",
            "previous_request_ids": ["req-previous"],
            "timestamps": True,
            "output_format": "mp3_44100_128",
            "output_path": str(out),
        }
    )
    assert result.success, result.error
    assert post.call_args.args[0].endswith("/with-timestamps")
    payload = post.call_args.kwargs["json"]
    assert payload["previous_text"] == "Earlier narration."
    assert payload["previous_request_ids"] == ["req-previous"]
    assert payload["voice_settings"]["use_speaker_boost"] is True
    assert result.data["selected_tool"] == "elevenlabs_tts"
    assert result.data["request_id"] == "req-01"
    assert [w["word"] for w in result.data["word_timestamps"]] == text.split()
    sidecar = out.with_name(out.name + ".alignment.json")
    assert str(sidecar) in result.artifacts
    assert json.loads(sidecar.read_text(encoding="utf-8"))["words"] == result.data["word_timestamps"]
    assert result.data["alignment_path"] == str(sidecar)
    from tools.subtitle.subtitle_gen import SubtitleGen

    subtitle_path = tmp_path / "captions.srt"
    subtitles = SubtitleGen().execute(
        {
            "segments": [{"words": result.data["word_timestamps"]}],
            "format": "srt",
            "output_path": str(subtitle_path),
        }
    )
    assert subtitles.success
    caption_text = subtitle_path.read_text(encoding="utf-8")
    assert "One" in caption_text and "consistent" in caption_text


@pytest.mark.parametrize("model", ["eleven_v3", "eleven_v4_turbo"])
def test_unsupported_models_reject_stitching_before_request(monkeypatch, model):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "unit-test-secret")
    post = Mock()
    monkeypatch.setattr("requests.post", post)
    result = ElevenLabsTTS().execute(
        {"text": "hello", "model_id": model, "previous_request_ids": ["r1"]}
    )
    assert not result.success
    assert "not supported" in result.error or "does not support" in result.error
    post.assert_not_called()


def test_missing_credentials_and_api_failure_never_expose_credentials(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    missing = ElevenLabsTTS().execute({"text": "hello"})
    assert not missing.success
    assert "ELEVENLABS_API_KEY" in missing.error

    secret = "never-print-this-secret"
    monkeypatch.setenv("ELEVENLABS_API_KEY", secret)
    monkeypatch.setattr("requests.post", Mock(side_effect=RuntimeError(f"failed with {secret}")))
    failed = ElevenLabsTTS().execute({"text": "hello"})
    assert not failed.success
    assert secret not in failed.error
    assert "[REDACTED]" in failed.error


def test_unsupported_model_fails_without_request(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "unit-test-secret")
    post = Mock()
    monkeypatch.setattr("requests.post", post)
    result = ElevenLabsTTS().execute({"text": "hello", "model_id": "future-model"})
    assert not result.success
    post.assert_not_called()


def test_selector_does_not_switch_after_selected_provider_failure(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "unit-test-secret")
    direct = ElevenLabsTTS()
    fallback = GoogleTTS()
    selector = TTSSelector()
    monkeypatch.setattr(selector, "_providers", lambda: [direct, fallback])
    monkeypatch.setattr(fallback, "get_status", lambda: ToolStatus.AVAILABLE)
    calls = []
    monkeypatch.setattr(direct, "execute", lambda _inputs: (calls.append("direct") or ToolResult(success=False, error="provider failed")))
    monkeypatch.setattr(fallback, "execute", lambda _inputs: (calls.append("google") or ToolResult(success=True)))
    result = selector.execute({"text": "hello", "preferred_tool": "elevenlabs_tts"})
    assert not result.success
    assert calls == ["direct"]


def test_explicit_tool_filter_and_existing_local_cloud_providers_remain_selectable():
    from tools.provider_routing import filter_explicit_route

    direct, google, piper = ElevenLabsTTS(), GoogleTTS(), PiperTTS()
    assert [t.name for t in filter_explicit_route({"preferred_tool": "elevenlabs_tts"}, [direct, google, piper])] == ["elevenlabs_tts"]
    assert google.capability == piper.capability == "tts"
    assert google.provider == "google_tts" and piper.provider == "piper"
    assert piper.estimate_cost({"text": "offline"}) == 0.0


def test_alignment_rejects_mismatched_timing_arrays():
    malformed = _alignment("Hi")
    malformed["character_end_times_seconds"].pop()
    with pytest.raises(ValueError, match="mismatched"):
        ElevenLabsTTS._validate_alignment(malformed)
