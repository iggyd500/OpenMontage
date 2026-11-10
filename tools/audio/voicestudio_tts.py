"""VoiceStudio's cached local TTS backend, exposed through the TTS registry."""

from __future__ import annotations

import io
import threading
import time
import wave
from pathlib import Path
from typing import Any

import requests

from lib.config_model import OpenMontageConfig, VoiceStudioTTSConfig
from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


# VoiceStudio 0.5.7's memory manager serializes heavyweight TTS engines. Keep
# requests from this process serialized too, so parallel pipeline calls do not
# compete for the workstation's limited RAM.
_GENERATION_LOCK = threading.Lock()


class VoiceStudioTTS(BaseTool):
    """Generate WAV narration through VoiceStudio's local OpenAI-compatible API.

    This adapter intentionally supports only the cached KittenTTS Mini 0.8
    model. It checks VoiceStudio's reported cache size before synthesis and
    never launches VoiceStudio or makes a request that could download weights.
    """

    name = "voicestudio_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "voicestudio"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = [
        "VoiceStudio 0.5.7+ backend running on the configured local endpoint",
        "KittenML/kitten-tts-mini-0.8 weights already cached in VoiceStudio",
    ]
    install_instructions = (
        "Start VoiceStudio and confirm its backend is healthy. Ensure KittenTTS "
        "Mini 0.8 is already cached in VoiceStudio. OpenMontage will not launch "
        "VoiceStudio or download model weights."
    )
    agent_skills = ["voicestudio-tts"]

    capabilities = ["text_to_speech", "voice_selection", "offline_generation"]
    supports = {
        "voice_cloning": False,
        "multilingual": False,
        "offline": True,
        "native_audio": True,
        "concurrent_requests": False,
    }
    best_for = [
        "local English narration with KittenTTS preset voices",
        "Jasper voice when its preferred tone is worth slower CPU rendering",
    ]
    not_good_for = [
        "high-throughput parallel narration on memory-constrained machines",
        "languages other than English or voice cloning",
    ]

    _VOICE_ALIASES = {
        "bella": "expr-voice-2-f",
        "jasper": "expr-voice-2-m",
        "luna": "expr-voice-3-f",
        "bruno": "expr-voice-3-m",
        "rosie": "expr-voice-4-f",
        "hugo": "expr-voice-4-m",
        "kiki": "expr-voice-5-f",
        "leo": "expr-voice-5-m",
    }
    _VOICE_IDS = tuple(_VOICE_ALIASES.values())
    _MODEL = "kittentts"
    _STATUS_TIMEOUT_SECONDS = 1.5
    _CACHE_CHECK_TIMEOUT_SECONDS = 10.0

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Narration text to synthesize."},
            "model": {
                "type": "string",
                "enum": [_MODEL],
                "default": _MODEL,
                "description": "VoiceStudio engine ID. This adapter supports KittenTTS only.",
            },
            "voice": {
                "type": "string",
                "enum": ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo", *_VOICE_IDS],
                "description": "KittenTTS 0.8 preset name or preset ID; Jasper maps to expr-voice-2-m.",
            },
            "voice_id": {
                "type": "string",
                "description": "Alias for the KittenTTS preset voice.",
            },
            "speed": {
                "type": "number",
                "minimum": 0.25,
                "maximum": 4.0,
                "description": "VoiceStudio speaking speed; 1.0 is normal speed.",
            },
            "voice_speed": {"type": "number", "minimum": 0.5, "maximum": 2.0},
            "speaking_rate": {"type": "number", "minimum": 0.25, "maximum": 2.0},
            "language": {"type": "string", "enum": ["en"], "default": "en"},
            "output_format": {
                "type": "string",
                "enum": ["wav"],
                "default": "wav",
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2,
        ram_mb=4096,
        vram_mb=0,
        disk_mb=0,
        network_required=False,
    )
    retry_policy = RetryPolicy(max_retries=0, retryable_errors=[])
    idempotency_key_fields = ["text", "model", "voice", "speed"]
    side_effects = ["writes a validated WAV file", "sends text to the local VoiceStudio backend"]
    user_visible_verification = ["Listen to the WAV and review pronunciation and pacing"]

    def __init__(self, config: VoiceStudioTTSConfig | None = None) -> None:
        self.config = config or OpenMontageConfig.load().tts.voicestudio
        self._health_cache_at = 0.0
        self._health_cache: tuple[ToolStatus, str] | None = None
        self._health_lock = threading.Lock()

    @property
    def _base_url(self) -> str:
        return self.config.base_url.rstrip("/")

    def _check_health(self, *, force: bool = False) -> tuple[ToolStatus, str]:
        now = time.monotonic()
        with self._health_lock:
            if (
                not force
                and self._health_cache is not None
                and now - self._health_cache_at < 1.0
            ):
                return self._health_cache

            try:
                response = requests.get(
                    f"{self._base_url}/health",
                    timeout=self._STATUS_TIMEOUT_SECONDS,
                )
            except requests.exceptions.Timeout:
                result = (
                    ToolStatus.UNAVAILABLE,
                    "VoiceStudio health check timed out.",
                )
            except requests.exceptions.RequestException:
                result = (
                    ToolStatus.UNAVAILABLE,
                    "VoiceStudio backend is not responding.",
                )
            else:
                if response.status_code == 200:
                    result = (ToolStatus.AVAILABLE, "VoiceStudio backend is healthy.")
                else:
                    result = (
                        ToolStatus.UNAVAILABLE,
                        f"VoiceStudio health endpoint returned HTTP {response.status_code}.",
                    )

            self._health_cache_at = now
            self._health_cache = result
            return result

    def get_status(self) -> ToolStatus:
        return self._check_health()[0]

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def _failure(self, message: str, started: float) -> ToolResult:
        return ToolResult(
            success=False,
            error=message,
            cost_usd=0.0,
            duration_seconds=round(time.perf_counter() - started, 2),
            model=self.config.model,
        )

    def _cached_model_bytes(self) -> tuple[int | None, str | None]:
        """Return the installed Kitten model bytes or an explanation.

        This is a read-only VoiceStudio endpoint. A missing/unknown cache state
        prevents the speech request, whose first use could otherwise download
        model weights.
        """
        try:
            response = requests.get(
                f"{self._base_url}/engines/kittentts/disk-usage",
                timeout=min(self.config.timeout, self._CACHE_CHECK_TIMEOUT_SECONDS),
            )
        except requests.exceptions.Timeout:
            return None, "Timed out while checking the cached KittenTTS model."
        except requests.exceptions.RequestException:
            return None, "Could not check the cached KittenTTS model in VoiceStudio."

        if response.status_code != 200:
            return (
                None,
                f"VoiceStudio model-cache check returned HTTP {response.status_code}; no speech request was sent.",
            )
        try:
            actual = response.json().get("actual", {})
            model_bytes = actual.get("model_bytes")
            model_bytes = int(model_bytes) if model_bytes is not None else None
        except (AttributeError, TypeError, ValueError):
            return None, "VoiceStudio returned an unreadable model-cache status; no speech request was sent."

        if not model_bytes or model_bytes < 1:
            return (
                0,
                "KittenTTS Mini 0.8 weights are not cached in VoiceStudio. Cache the model in VoiceStudio first; OpenMontage did not request audio and will not download it.",
            )
        return model_bytes, None

    @classmethod
    def _canonical_voice(cls, value: str) -> str | None:
        value = value.strip()
        if value in cls._VOICE_IDS:
            return value
        return cls._VOICE_ALIASES.get(value.casefold())

    @staticmethod
    def _inspect_wav(audio: bytes) -> tuple[float, int, int] | None:
        try:
            with wave.open(io.BytesIO(audio), "rb") as source:
                sample_rate = source.getframerate()
                channels = source.getnchannels()
                frames = source.getnframes()
                if sample_rate <= 0 or channels <= 0 or frames <= 0:
                    return None
                return frames / sample_rate, sample_rate, channels
        except (EOFError, OSError, wave.Error):
            return None

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        text = inputs.get("text")
        if not isinstance(text, str) or not text.strip():
            return self._failure("VoiceStudio TTS requires non-empty text.", started)

        model = inputs.get("model", self.config.model)
        if model != self._MODEL:
            return self._failure(
                "VoiceStudio TTS currently supports only the 'kittentts' model.",
                started,
            )

        requested_voice = (
            inputs.get("voice")
            or inputs.get("voice_id")
            or self.config.voice
        )
        voice = self._canonical_voice(str(requested_voice))
        if voice is None:
            return self._failure(
                "Unsupported KittenTTS voice. Use a KittenTTS 0.8 preset name or preset ID.",
                started,
            )

        speed_value = inputs.get(
            "speed",
            inputs.get("voice_speed", inputs.get("speaking_rate", self.config.speed)),
        )
        try:
            speed = float(speed_value)
        except (TypeError, ValueError):
            return self._failure("VoiceStudio speed must be a number.", started)
        if not 0.25 <= speed <= 4.0:
            return self._failure(
                "VoiceStudio speed must be between 0.25 and 4.0.", started
            )

        language = inputs.get("language", "en")
        if language not in (None, "en", "en-US"):
            return self._failure(
                "VoiceStudio KittenTTS supports English only (language='en').",
                started,
            )

        requested_format = inputs.get("output_format", "wav")
        if str(requested_format).casefold() not in {"wav", "riff"}:
            return self._failure(
                "VoiceStudio TTS currently returns WAV audio only.", started
            )

        # A single lock covers cache verification and synthesis. No concurrent
        # requests reach VoiceStudio from this process.
        with _GENERATION_LOCK:
            health_status, health_message = self._check_health(force=True)
            if health_status != ToolStatus.AVAILABLE:
                return self._failure(
                    f"{health_message} Ensure the local VoiceStudio backend is running at {self._base_url}; OpenMontage will not launch it.",
                    started,
                )

            model_bytes, cache_error = self._cached_model_bytes()
            if cache_error:
                return self._failure(cache_error, started)

            payload = {
                "model": self._MODEL,
                "input": text,
                "voice": voice,
                "language": "en",
                "speed": speed,
                "response_format": "wav",
            }
            try:
                response = requests.post(
                    f"{self._base_url}/v1/audio/speech",
                    json=payload,
                    timeout=self.config.timeout,
                )
            except requests.exceptions.Timeout:
                return self._failure(
                    f"VoiceStudio TTS timed out after {self.config.timeout:g} seconds.",
                    started,
                )
            except requests.exceptions.ConnectionError:
                with self._health_lock:
                    self._health_cache = (
                        ToolStatus.UNAVAILABLE,
                        "VoiceStudio backend is not responding.",
                    )
                    self._health_cache_at = time.monotonic()
                return self._failure(
                    f"VoiceStudio backend is not responding at {self._base_url}. Start VoiceStudio and retry; OpenMontage will not launch it.",
                    started,
                )
            except requests.exceptions.RequestException as exc:
                return self._failure(
                    f"VoiceStudio TTS request failed ({type(exc).__name__}).",
                    started,
                )

            if response.status_code != 200:
                return self._failure(
                    f"VoiceStudio TTS returned HTTP {response.status_code}; no provider fallback was attempted.",
                    started,
                )

            audio = response.content
            wav_info = self._inspect_wav(audio)
            if wav_info is None:
                return self._failure(
                    "VoiceStudio returned an invalid or empty WAV file.", started
                )
            duration, sample_rate, channels = wav_info

            output_path = Path(
                inputs.get("output_path") or "tts_voicestudio.wav"
            )
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(audio)
            except OSError as exc:
                return self._failure(
                    f"Could not save the VoiceStudio WAV ({type(exc).__name__}).",
                    started,
                )

        elapsed = round(time.perf_counter() - started, 2)
        voice_name = next(
            (name for name, preset in self._VOICE_ALIASES.items() if preset == voice),
            voice,
        )
        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": self._MODEL,
                "voice": voice,
                "voice_name": voice_name.title(),
                "speed": speed,
                "language": "en",
                "format": "wav",
                "output": str(output_path),
                "duration_seconds": round(duration, 3),
                "generation_seconds": elapsed,
                "sample_rate": sample_rate,
                "channels": channels,
                "model_cache_bytes": model_bytes,
            },
            artifacts=[str(output_path)],
            cost_usd=0.0,
            duration_seconds=elapsed,
            model=self._MODEL,
        )
