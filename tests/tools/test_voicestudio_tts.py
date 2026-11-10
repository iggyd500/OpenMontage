from __future__ import annotations

import io
import threading
import time
import wave
from unittest.mock import Mock

import pytest
import requests

from lib.config_model import OpenMontageConfig, VoiceStudioTTSConfig
from tools.base_tool import ToolResult, ToolStatus
from tools.audio.piper_tts import PiperTTS
from tools.audio.tts_selector import TTSSelector
from tools.audio.voicestudio_tts import VoiceStudioTTS
from tools.tool_registry import ToolRegistry


def _wav_bytes(*, duration_seconds: float = 0.5, sample_rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(b"\0\0" * int(duration_seconds * sample_rate))
    return buffer.getvalue()


def _response(status_code: int = 200, content: bytes = b"", json_data=None):
    response = Mock()
    response.status_code = status_code
    response.content = content
    response.json.return_value = json_data
    return response


def _mock_ready_backend(monkeypatch, *, wav: bytes | None = None, model_bytes: int = 81_547_388):
    monkeypatch.setattr(
        "tools.audio.voicestudio_tts.requests.get",
        Mock(
            side_effect=[
                _response(),
                _response(json_data={"actual": {"model_bytes": model_bytes}}),
            ]
        ),
    )
    post = Mock(return_value=_response(content=wav or _wav_bytes()))
    monkeypatch.setattr("tools.audio.voicestudio_tts.requests.post", post)
    return post


def test_default_config_and_jasper_preset_mapping():
    config = OpenMontageConfig.load()
    assert config.tts.provider == "elevenlabs"
    assert config.tts.elevenlabs.model_id == "eleven_multilingual_v2"
    assert config.tts.elevenlabs.voice_name == "Brian"
    assert config.tts.elevenlabs.voice_id == "nPczCjzI2devNBz1zQrb"
    assert config.tts.elevenlabs.language == "en"
    assert config.tts.elevenlabs.output_format == "wav_44100"
    assert config.tts.voicestudio.model == "kittentts"
    assert config.tts.voicestudio.voice == "expr-voice-2-m"
    assert VoiceStudioTTS._canonical_voice("Jasper") == "expr-voice-2-m"
    assert VoiceStudioTTS._canonical_voice("expr-voice-3-m") == "expr-voice-3-m"


def test_config_rejects_invalid_endpoint_voice_speed_and_timeout():
    for values in (
        {"base_url": "ftp://localhost:3900"},
        {"base_url": "http://user:pass@localhost:3900"},
        {"voice": "expr-voice-3-missing"},
        {"speed": 0},
        {"timeout": 1801},
    ):
        with pytest.raises(ValueError):
            VoiceStudioTTSConfig(**values)


def test_generates_valid_wav_and_returns_provider_metadata(tmp_path, monkeypatch):
    out = tmp_path / "assets" / "narration.wav"
    post = _mock_ready_backend(monkeypatch)

    result = VoiceStudioTTS().execute(
        {"text": "Short narration.", "voice": "Jasper", "output_path": str(out)}
    )

    assert result.success, result.error
    assert result.data["provider"] == "voicestudio"
    assert result.data["model"] == "kittentts"
    assert result.data["voice"] == "expr-voice-2-m"
    assert result.data["voice_name"] == "Jasper"
    assert result.data["duration_seconds"] == 0.5
    assert result.data["sample_rate"] == 24000
    assert result.data["channels"] == 1
    assert result.data["model_cache_bytes"] == 81_547_388
    assert out.read_bytes() == _wav_bytes()
    assert post.call_args.kwargs["json"]["voice"] == "expr-voice-2-m"
    assert post.call_args.kwargs["timeout"] == 900


def test_missing_model_cache_refuses_to_post(tmp_path, monkeypatch):
    get = Mock(
        side_effect=[
            _response(),
            _response(json_data={"actual": {"model_bytes": 0}}),
        ]
    )
    post = Mock()
    monkeypatch.setattr("tools.audio.voicestudio_tts.requests.get", get)
    monkeypatch.setattr("tools.audio.voicestudio_tts.requests.post", post)

    result = VoiceStudioTTS().execute(
        {"text": "No model download", "output_path": str(tmp_path / "never.wav")}
    )

    assert not result.success
    assert "did not request audio" in result.error
    post.assert_not_called()
    assert not (tmp_path / "never.wav").exists()


def test_backend_unavailable_and_http_errors_do_not_fall_back(monkeypatch):
    unavailable = VoiceStudioTTS()
    monkeypatch.setattr(
        "tools.audio.voicestudio_tts.requests.get",
        Mock(side_effect=requests.exceptions.ConnectionError("local backend down")),
    )
    result = unavailable.execute({"text": "Hello"})
    assert not result.success
    assert "VoiceStudio backend" in result.error

    voice = VoiceStudioTTS()
    monkeypatch.setattr(
        "tools.audio.voicestudio_tts.requests.get",
        Mock(
            side_effect=[
                _response(),
                _response(json_data={"actual": {"model_bytes": 100}}),
            ]
        ),
    )
    monkeypatch.setattr(
        "tools.audio.voicestudio_tts.requests.post", Mock(return_value=_response(503))
    )
    result = voice.execute({"text": "HTTP error"})
    assert not result.success
    assert "HTTP 503" in result.error
    assert "no provider fallback" in result.error


def test_post_timeout_and_invalid_wav_are_reported_safely(monkeypatch):
    _mock_ready_backend(monkeypatch)
    monkeypatch.setattr(
        "tools.audio.voicestudio_tts.requests.post",
        Mock(side_effect=requests.exceptions.Timeout()),
    )
    timed_out = VoiceStudioTTS().execute({"text": "Wait"})
    assert not timed_out.success
    assert "timed out" in timed_out.error

    _mock_ready_backend(monkeypatch, wav=b"not-a-wave")
    invalid = VoiceStudioTTS().execute({"text": "Bad payload"})
    assert not invalid.success
    assert "invalid or empty WAV" in invalid.error


def test_voice_studio_generation_requests_are_serialized(monkeypatch, tmp_path):
    active = 0
    maximum = 0
    guard = threading.Lock()

    def get(url, **_kwargs):
        if url.endswith("/health"):
            return _response()
        return _response(json_data={"actual": {"model_bytes": 100}})

    def post(*_args, **_kwargs):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with guard:
            active -= 1
        return _response(content=_wav_bytes())

    monkeypatch.setattr("tools.audio.voicestudio_tts.requests.get", get)
    monkeypatch.setattr("tools.audio.voicestudio_tts.requests.post", post)
    tool = VoiceStudioTTS()
    results = []

    def generate(index):
        results.append(
            tool.execute({"text": f"Section {index}", "output_path": str(tmp_path / f"{index}.wav")})
        )

    workers = [threading.Thread(target=generate, args=(index,)) for index in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=3)

    assert all(result.success for result in results)
    assert maximum == 1


def test_registry_discovers_voicestudio_without_network():
    registry = ToolRegistry()
    names = registry.discover()
    assert "voicestudio_tts" in names
    tool = registry.get("voicestudio_tts")
    assert tool is not None
    assert tool.provider == "voicestudio"
    assert tool.capability == "tts"


def test_selector_explicit_voice_studio_route_and_strict_provider_default(monkeypatch):
    voice = VoiceStudioTTS()
    piper = PiperTTS()
    selector = TTSSelector()
    monkeypatch.setattr(selector, "_providers", lambda: [voice, piper])
    monkeypatch.setattr(voice, "get_status", lambda: ToolStatus.AVAILABLE)
    monkeypatch.setattr(piper, "get_status", lambda: ToolStatus.AVAILABLE)
    called = []
    monkeypatch.setattr(
        voice,
        "execute",
        lambda inputs: (called.append(("voice", inputs)) or ToolResult(success=True, data={"provider": "voicestudio"})),
    )
    monkeypatch.setattr(
        piper,
        "execute",
        lambda inputs: ToolResult(success=True, data={"provider": "piper"}),
    )
    selected = selector.execute({"text": "hello", "preferred_tool": "voicestudio_tts"})
    assert selected.success
    assert selected.data["selected_tool"] == "voicestudio_tts"
    assert called and called[0][0] == "voice"

    defaulted = selector.execute({"text": "hello"})
    assert not defaulted.success
    assert "Configured TTS provider 'elevenlabs' is unavailable" in defaulted.error

    monkeypatch.setattr(voice, "get_status", lambda: ToolStatus.UNAVAILABLE)
    failed = selector.execute({"text": "hello", "preferred_tool": "voicestudio_tts"})
    assert not failed.success
    assert "no provider fallback" in failed.error


def test_existing_piper_google_and_elevenlabs_are_discoverable():
    registry = ToolRegistry()
    names = set(registry.discover())
    assert {"piper_tts", "google_tts", "elevenlabs_tts", "voicestudio_tts"} <= names

