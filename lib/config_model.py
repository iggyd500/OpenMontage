"""Runtime configuration model for OpenMontage.

Loads config.yaml, merges with env overrides, and provides typed access.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class BudgetMode(str, Enum):
    OBSERVE = "observe"
    WARN = "warn"
    CAP = "cap"


class CheckpointPolicy(str, Enum):
    GUIDED = "guided"
    MANUAL_ALL = "manual_all"
    AUTO_NONCREATIVE = "auto_noncreative"


class LLMConfig(BaseModel):
    provider: Literal["gemini", "ollama"] = "gemini"
    model: str = "gemini-3.5-flash-lite"
    temperature: float = 0.7
    max_tokens: int = 4096
    ollama_base_url: str = "http://localhost:11434"
    request_timeout_seconds: float = 120.0

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("llm.model must be a non-empty model identifier")
        return value

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if not 0 <= value <= 2:
            raise ValueError("llm.temperature must be between 0 and 2")
        return value

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int) -> int:
        if value < 1:
            raise ValueError("llm.max_tokens must be greater than zero")
        return value

    @field_validator("request_timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if not 0 < value <= 600:
            raise ValueError("llm.request_timeout_seconds must be in (0, 600]")
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("llm.ollama_base_url must be an HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("llm.ollama_base_url must not contain credentials")
        return value

    @model_validator(mode="after")
    def validate_provider_model_pair(self) -> "LLMConfig":
        if self.provider == "gemini" and not self.model.startswith("gemini-"):
            raise ValueError("Gemini provider requires a Gemini model identifier")
        if self.provider == "ollama" and self.model.startswith("gemini-"):
            raise ValueError("Ollama provider requires an Ollama model tag")
        return self


class VoiceStudioTTSConfig(BaseModel):
    """Settings for the optional local VoiceStudio speech backend."""

    base_url: str = "http://localhost:3900"
    model: Literal["kittentts"] = "kittentts"
    voice: str = "expr-voice-2-m"
    speed: float = 1.0
    timeout: float = 900.0

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("tts.voicestudio.base_url must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "tts.voicestudio.base_url must not contain credentials, query, or fragment"
            )
        return value

    @field_validator("voice")
    @classmethod
    def validate_voice(cls, value: str) -> str:
        allowed = {
            "expr-voice-2-f",
            "expr-voice-2-m",
            "expr-voice-3-f",
            "expr-voice-3-m",
            "expr-voice-4-f",
            "expr-voice-4-m",
            "expr-voice-5-f",
            "expr-voice-5-m",
        }
        value = value.strip()
        if value not in allowed:
            raise ValueError(
                "tts.voicestudio.voice must be a KittenTTS 0.8 preset ID"
            )
        return value

    @field_validator("speed")
    @classmethod
    def validate_speed(cls, value: float) -> float:
        if not 0.25 <= value <= 4.0:
            raise ValueError("tts.voicestudio.speed must be between 0.25 and 4.0")
        return value

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if not 0 < value <= 1800:
            raise ValueError("tts.voicestudio.timeout must be in (0, 1800]")
        return value


class ElevenLabsTTSConfig(BaseModel):
    """Preferred direct ElevenLabs narration settings."""

    model_id: str = "eleven_multilingual_v2"
    voice_name: str = "Brian"
    voice_id: str = "nPczCjzI2devNBz1zQrb"
    language: Literal["en"] = "en"
    output_format: Literal["wav_44100"] = "wav_44100"

    @field_validator("model_id", "voice_name", "voice_id")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ElevenLabs TTS settings must not be empty")
        return value


class TTSConfig(BaseModel):
    """Default TTS provider and provider-specific settings."""

    provider: str = "elevenlabs"
    elevenlabs: ElevenLabsTTSConfig = Field(default_factory=ElevenLabsTTSConfig)
    voicestudio: VoiceStudioTTSConfig = Field(default_factory=VoiceStudioTTSConfig)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("tts.provider must be a non-empty provider ID")
        return value


class BudgetConfig(BaseModel):
    mode: BudgetMode = BudgetMode.WARN
    total_usd: float = 10.0
    reserve_pct: float = 0.10
    single_action_approval_usd: float = 0.50
    require_approval_for_new_paid_tool: bool = True


class CheckpointConfig(BaseModel):
    policy: CheckpointPolicy = CheckpointPolicy.GUIDED
    storage_dir: str = "pipeline"


class OutputConfig(BaseModel):
    default_format: str = "mp4"
    default_codec: str = "libx264"
    default_audio_codec: str = "aac"
    default_resolution: str = "1920x1080"
    default_fps: int = 30
    default_crf: int = 23


class PathsConfig(BaseModel):
    pipeline_dir: str = "pipeline"
    library_dir: str = "library"
    styles_dir: str = "styles"
    skills_dir: str = "skills"
    output_dir: str = "output"


class OpenMontageConfig(BaseModel):
    """Top-level runtime configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "OpenMontageConfig":
        """Load config from YAML file. Falls back to defaults if file missing."""
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "config.yaml"

        if config_path.exists():
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
            return cls.model_validate(raw)

        return cls()

    def resolve_path(self, key: str, project_root: Optional[Path] = None) -> Path:
        """Resolve a relative path from PathsConfig against project root."""
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent
        value = getattr(self.paths, key)
        return (project_root / value).resolve()
