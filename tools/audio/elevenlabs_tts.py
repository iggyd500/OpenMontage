"""ElevenLabs text-to-speech provider tool."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from tools.provider_pricing import PriceQuoteRequired

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


class ElevenLabsTTS(BaseTool):
    name = "elevenlabs_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "elevenlabs"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set the ELEVENLABS_API_KEY environment variable:\n"
        "  export ELEVENLABS_API_KEY=your_key_here\n"
        "Get a key at https://elevenlabs.io\n"
        "If fal_elevenlabs_tts is available, use it instead to access ElevenLabs "
        "speech through fal.ai without a separate ElevenLabs key."
    )
    fallback = "openai_tts"
    fallback_tools = ["openai_tts", "piper_tts"]
    agent_skills = ["elevenlabs", "text-to-speech"]

    capabilities = [
        "text_to_speech",
        "voice_selection",
        "ssml_support",
        "pronunciation_control",
    ]
    supports = {
        "voice_cloning": True,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
    }
    best_for = [
        "high-quality narration",
        "voice-sensitive spokesperson videos",
        "multilingual spoken delivery",
    ]
    not_good_for = [
        "fully offline production",
        "privacy-constrained local-only workflows",
    ]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Text to convert to speech"},
            "voice_id": {
                "type": "string",
                "description": "ElevenLabs voice ID (OpenMontage default: Brian)",
            },
            "voice_name": {
                "type": "string",
                "description": "Optional display name for the selected voice.",
            },
            "language": {
                "type": "string",
                "enum": ["en"],
                "description": (
                    "OpenMontage language metadata. Multilingual v2 detects language "
                    "from the supplied text and does not accept language_code."
                ),
            },
            "model_id": {
                "type": "string",
                "default": "eleven_multilingual_v2",
                "enum": [
                    "eleven_v4",
                    "eleven_v4_turbo",
                    "eleven_v3",
                    "eleven_multilingual_v2",
                    "eleven_turbo_v2_5",
                    "eleven_flash_v2_5",
                ],
                "description": "TTS model to use",
            },
            "stability": {
                "type": "number",
                "default": 0.5,
                "minimum": 0,
                "maximum": 1,
            },
            "similarity_boost": {
                "type": "number",
                "default": 0.75,
                "minimum": 0,
                "maximum": 1,
            },
            "style": {
                "type": "number",
                "default": 0.0,
                "minimum": 0,
                "maximum": 1,
            },
            "speed": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.7,
                "maximum": 1.2,
            },
            "use_speaker_boost": {
                "type": "boolean",
                "default": True,
            },
            "seed": {"type": "integer", "minimum": 0, "maximum": 4294967295},
            "language_code": {"type": "string"},
            "previous_text": {"type": "string"},
            "next_text": {"type": "string"},
            "previous_request_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
            },
            "next_request_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
            },
            "apply_text_normalization": {
                "type": "string",
                "enum": ["auto", "on", "off"],
            },
            "output_path": {"type": "string"},
            "output_format": {
                "type": "string",
                "default": "mp3_44100_128",
                "enum": [
                    "wav_44100",
                    "mp3_44100_128",
                    "mp3_44100_192",
                    "pcm_16000",
                    "pcm_24000",
                ],
                "description": (
                    "wav_44100 returns PCM WAV at 44.1 kHz, converted locally from "
                    "the standard MP3 44.1 kHz API response."
                ),
            },
            "with_timestamps": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Use the /with-timestamps endpoint and return word-level timings "
                    "(data.word_timestamps) plus the raw character alignment, saved next to "
                    "the audio as <output>.alignment.json. Use for caption sync and "
                    "word-cued animation without a separate transcription pass."
                ),
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=0)
    idempotency_key_fields = [
        "text",
        "voice_id",
        "model_id",
        "stability",
        "similarity_boost",
        "style",
        "speed",
        "use_speaker_boost",
    ] + [
        "seed",
        "language_code",
        "previous_text",
        "next_text",
        "previous_request_ids",
        "next_request_ids",
        "output_format",
        "with_timestamps",
    ]
    side_effects = ["writes audio file to output_path", "calls ElevenLabs API"]
    user_visible_verification = ["Listen to generated audio for natural speech quality"]

    DEFAULT_VOICE_ID = "nPczCjzI2devNBz1zQrb"
    DEFAULT_VOICE_NAME = "Brian"
    MODEL_CHARACTER_LIMITS = {
        "eleven_v4": 2_000,  # Keep the conservative Text-to-Dialogue limit.
        "eleven_v4_turbo": 2_000,
        "eleven_v3": 5_000,
        "eleven_multilingual_v2": 10_000,
        "eleven_turbo_v2_5": 40_000,
        "eleven_flash_v2_5": 40_000,
    }
    # Approximate public API rates per 1,000 characters; plan credits, taxes,
    # discounts and account-specific pricing can change the actual charge.
    MODEL_COST_PER_1000_CHARS = {
        "eleven_v3": 0.08,
        "eleven_multilingual_v2": 0.08,
        "eleven_turbo_v2_5": 0.04,
        "eleven_flash_v2_5": 0.04,
    }

    def get_status(self) -> ToolStatus:
        if os.environ.get("ELEVENLABS_API_KEY"):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        model_id = inputs.get("model_id") or "eleven_multilingual_v2"
        if model_id in {"eleven_v4", "eleven_v4_turbo"}:
            raise PriceQuoteRequired("Quote Eleven v4 pricing for your account plan")
        if model_id not in self.MODEL_CHARACTER_LIMITS:
            raise ValueError(f"Unsupported ElevenLabs model: {model_id}")
        if model_id not in self.MODEL_COST_PER_1000_CHARS:
            raise PriceQuoteRequired(
                f"No current estimate is configured for {model_id}; obtain a plan-specific quote"
            )
        rate = self.MODEL_COST_PER_1000_CHARS[model_id]
        return round(len(inputs.get("text", "")) * rate / 1000, 6)

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            return ToolResult(
                success=False,
                error="No ElevenLabs API key. " + self.install_instructions,
            )

        start = time.time()
        try:
            result = self._generate(inputs, api_key)
        except Exception as exc:
            message = str(exc)
            # Never include the API key, even if an HTTP/client exception echoes
            # request headers or a URL that accidentally contains credentials.
            message = message.replace(api_key, "[REDACTED]")
            message = re.sub(
                r"(?i)(xi-api-key\s*[:=]\s*)[^\s,;]+", r"\1[REDACTED]", message
            )
            return ToolResult(success=False, error=f"TTS generation failed: {message}")

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = (
            None
            if inputs.get("model_id", "").startswith("eleven_v4")
            else self.estimate_cost(inputs)
        )
        result.data["cost_status"] = (
            "unquoted" if result.cost_usd is None else "estimated"
        )
        result.data["cost_is_estimate"] = result.cost_usd is not None
        if result.cost_usd is not None:
            result.data["cost_estimate_usd"] = result.cost_usd
            result.data["cost_estimate_note"] = (
                "Estimate only; actual charges depend on account plan, included credits, and current pricing."
            )
        return result

    def _generate(self, inputs: dict[str, Any], api_key: str) -> ToolResult:
        import requests

        from jsonschema import validate

        validate(inputs, self.input_schema)
        text = inputs["text"]
        model_id = inputs.get("model_id", "eleven_multilingual_v2")
        limit = self.MODEL_CHARACTER_LIMITS.get(model_id)
        if limit is None:
            raise ValueError("Unsupported ElevenLabs model")
        if not text.strip() or len(text) > limit:
            raise ValueError(f"Text must contain 1–{limit} characters for this model")
        voice_id = inputs.get("voice_id", self.DEFAULT_VOICE_ID)
        output_format = inputs.get("output_format", "mp3_44100_128")
        wav_44100 = output_format == "wav_44100"
        api_output_format = "mp3_44100_128" if wav_44100 else output_format
        ext = "mp3" if output_format.startswith("mp3") else "wav"
        output_path = Path(inputs.get("output_path", f"tts_output.{ext}"))
        ffmpeg_executable = None
        if wav_44100:
            import shutil

            ffmpeg_executable = shutil.which("ffmpeg")
            if not ffmpeg_executable:
                raise RuntimeError(
                    "FFmpeg is required to produce local 44.1 kHz WAV output."
                )
            if output_path.exists():
                raise FileExistsError(
                    "Refusing to overwrite an existing ElevenLabs WAV output."
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
        voice_settings = {
            "stability": inputs.get("stability", 0.5),
            "similarity_boost": inputs.get("similarity_boost", 0.75),
            "style": inputs.get("style", 0.0),
            "speed": inputs.get("speed", 1.0),
            "use_speaker_boost": inputs.get("use_speaker_boost", True),
        }

        payload = {"text": text, "model_id": model_id, "voice_settings": voice_settings}
        continuity_fields = {
            "previous_text",
            "next_text",
            "previous_request_ids",
            "next_request_ids",
        }
        has_continuity = any(field in inputs for field in continuity_fields)
        stitching_models = {
            "eleven_v4",
            "eleven_multilingual_v2",
            "eleven_turbo_v2_5",
            "eleven_flash_v2_5",
        }
        if has_continuity and model_id not in stitching_models:
            raise ValueError(f"Request stitching is not supported for {model_id}")
        for field in (
            "seed",
            "language_code",
            "previous_text",
            "next_text",
            "previous_request_ids",
            "next_request_ids",
            "apply_text_normalization",
        ):
            if field in inputs:
                payload[field] = inputs[field]
        with_timestamps = bool(inputs.get("with_timestamps", False))
        endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        if with_timestamps:
            endpoint += "/with-timestamps"
        if model_id == "eleven_v4":
            unsupported = set(inputs) & {"style", "speed", "use_speaker_boost"}
            if unsupported:
                raise ValueError(
                    f"Eleven v4 dialogue does not support {sorted(unsupported)}"
                )
            payload = {
                "inputs": [{"text": text, "voice_id": voice_id}],
                "model_id": model_id,
                "settings": {
                    "stability": voice_settings["stability"],
                    "similarity": voice_settings["similarity_boost"],
                },
            }
            for field in (
                "seed",
                "language_code",
                "previous_text",
                "previous_request_ids",
                "next_request_ids",
                "apply_text_normalization",
            ):
                if field in inputs:
                    payload[field] = inputs[field]
            if "next_text" in inputs:
                payload["future_text"] = inputs["next_text"]
            if any(
                len(payload.get(k, "")) > 100 for k in ("previous_text", "future_text")
            ):
                raise ValueError("Dialogue context text must be at most 100 characters")
            endpoint = "https://api.elevenlabs.io/v1/text-to-dialogue"
            if with_timestamps:
                endpoint += "/with-timestamps"
        if model_id == "eleven_v4_turbo":
            audio_bytes = self._turbo_audio(
                inputs, api_key, voice_id, api_output_format
            )
            response = None
        else:
            response = requests.post(
                endpoint,
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json" if with_timestamps else "audio/mpeg",
                },
                json=payload,
                params={"output_format": api_output_format},
                timeout=120,
            )
            response.raise_for_status()
        alignment = None
        if model_id == "eleven_v4_turbo":
            pass
        elif with_timestamps:
            body = response.json()
            audio_bytes = base64.b64decode(body["audio_base64"])
            alignment = body.get("alignment") or body.get("normalized_alignment")
            if not alignment:
                raise ValueError(
                    "ElevenLabs returned no alignment for a with-timestamps request"
                )
            self._validate_alignment(alignment)
        else:
            audio_bytes = response.content

        if not audio_bytes:
            raise ValueError("ElevenLabs returned empty audio")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if wav_44100:
            import subprocess
            import tempfile
            import wave

            with tempfile.NamedTemporaryFile(
                prefix="openmontage-elevenlabs-",
                suffix=".mp3",
                dir=output_path.parent,
                delete=False,
            ) as source_file:
                source_path = Path(source_file.name)
                source_file.write(audio_bytes)
            try:
                conversion = subprocess.run(
                    [
                        ffmpeg_executable,
                        "-v",
                        "error",
                        "-n",
                        "-i",
                        str(source_path),
                        "-c:a",
                        "pcm_s16le",
                        "-ar",
                        "44100",
                        "-ac",
                        "1",
                        str(output_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=120,
                    check=False,
                )
                if conversion.returncode != 0:
                    output_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        "FFmpeg failed to convert ElevenLabs audio to WAV."
                    )
                with wave.open(str(output_path), "rb") as wav:
                    if wav.getframerate() != 44100 or wav.getnframes() <= 0:
                        output_path.unlink(missing_ok=True)
                        raise RuntimeError(
                            "FFmpeg produced an invalid 44.1 kHz ElevenLabs WAV."
                        )
            finally:
                source_path.unlink(missing_ok=True)
        elif output_format.startswith("pcm_"):
            import wave

            with wave.open(str(output_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(int(output_format.split("_")[1]))
                wav.writeframes(audio_bytes)
        else:
            output_path.write_bytes(audio_bytes)

        artifacts = [str(output_path)]
        extra: dict[str, Any] = {}
        if with_timestamps:
            words = self.words_from_alignment(alignment)
            try:
                from tools.analysis.audio_probe import probe_duration

                audio_duration = probe_duration(output_path)
            except Exception:
                audio_duration = None
            if audio_duration is not None and words[-1]["end"] > audio_duration + 0.05:
                raise ValueError(
                    "ElevenLabs alignment extends beyond the generated audio duration"
                )
            alignment_path = output_path.with_name(output_path.name + ".alignment.json")
            alignment_path.write_text(
                json.dumps(
                    {"text": text, "alignment": alignment, "words": words},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            artifacts.append(str(alignment_path))
            extra = {
                "word_timestamps": words,
                "alignment_path": str(alignment_path),
                "audio_duration_seconds": round(audio_duration, 3)
                if audio_duration is not None
                else None,
                # False when the service normalised the text (numbers, symbols); timings
                # then refer to the spoken form returned in the alignment.
                "alignment_matches_text": "".join(alignment["characters"]) == text,
            }

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": model_id,
                "voice_id": voice_id,
                "voice_name": inputs.get(
                    "voice_name",
                    self.DEFAULT_VOICE_NAME
                    if voice_id == self.DEFAULT_VOICE_ID
                    else None,
                ),
                "language": inputs.get("language", "en"),
                "voice_settings": voice_settings,
                "text_length": len(text),
                "output": str(output_path),
                "format": output_format,
                "api_output_format": api_output_format,
                "sample_rate": 44100 if wav_44100 else None,
                "request_id": (
                    response.headers.get("request-id") if response is not None else None
                ),
                **extra,
            },
            artifacts=artifacts,
            model=model_id,
        )

    @staticmethod
    def _turbo_audio(inputs, api_key, voice_id, output_format):
        # Current realtime-tdd tutorial explicitly supports v4 Turbo; the older
        # API reference still says v3-only. Do not route Turbo to legacy HTTP TTS.
        from urllib.parse import urlencode
        from websockets.sync.client import connect

        unsupported = set(inputs) & {
            "stability",
            "similarity_boost",
            "style",
            "speed",
            "use_speaker_boost",
            "previous_text",
            "next_text",
            "previous_request_ids",
            "next_request_ids",
        }
        if unsupported or inputs.get("with_timestamps"):
            raise ValueError(
                "Turbo file synthesis does not support voice settings, continuity or timestamps"
            )
        query = {"model_id": "eleven_v4_turbo", "output_format": output_format}
        for field in ("seed", "language_code", "apply_text_normalization"):
            if field in inputs:
                query[field] = inputs[field]
        chunks = []
        deadline = time.monotonic() + 180
        with connect(
            "wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input?"
            + urlencode(query),
            additional_headers={"xi-api-key": api_key},
            open_timeout=20,
        ) as socket:
            socket.send(json.dumps({"voices": [voice_id]}))
            socket.send(
                json.dumps(
                    {
                        "inputs": [
                            {
                                "text": inputs["text"],
                                "voice_id": voice_id,
                                "new_turn": False,
                            }
                        ]
                    }
                )
            )
            socket.send(json.dumps({"close_socket": True}))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Dialogue stream did not finish; do not automatically resubmit"
                    )
                event = json.loads(socket.recv(timeout=remaining))
                if event.get("error"):
                    raise ValueError("ElevenLabs dialogue stream returned an error")
                if event.get("audio"):
                    chunks.append(base64.b64decode(event["audio"], validate=True))
                if event.get("is_final"):
                    break
        return b"".join(chunks)

    @staticmethod
    def _validate_alignment(alignment: dict[str, Any]) -> None:
        """Reject malformed character timing arrays before sidecar creation."""
        required = (
            "characters",
            "character_start_times_seconds",
            "character_end_times_seconds",
        )
        if not isinstance(alignment, dict) or any(k not in alignment for k in required):
            raise ValueError("ElevenLabs returned malformed alignment data")
        chars, starts, ends = (alignment[k] for k in required)
        if (
            not isinstance(chars, list)
            or not isinstance(starts, list)
            or not isinstance(ends, list)
        ):
            raise ValueError("ElevenLabs returned malformed alignment data")
        if len(chars) != len(starts) or len(chars) != len(ends):
            raise ValueError("ElevenLabs returned mismatched alignment timing arrays")
        if not all(
            isinstance(t, (int, float)) and math.isfinite(t)
            for t in starts + ends
        ):
            raise ValueError("ElevenLabs returned non-numeric alignment timings")
        previous_start = previous_end = -1.0
        for start, end in zip(starts, ends):
            if start < 0 or end < start or start < previous_start or end < previous_end:
                raise ValueError(
                    "ElevenLabs returned negative, reversed, or non-chronological alignment timings"
                )
            previous_start, previous_end = start, end
        if not all(isinstance(char, str) for char in chars):
            raise ValueError("ElevenLabs returned malformed alignment characters")

    @staticmethod
    def words_from_alignment(alignment: dict[str, Any]) -> list[dict[str, Any]]:
        """Group ElevenLabs character timings into whitespace-delimited words.

        Each word gets the start of its first character and the end of its last one,
        in seconds from the start of the returned audio.
        """
        pieces = alignment["characters"]
        starts = alignment["character_start_times_seconds"]
        ends = alignment["character_end_times_seconds"]
        text, owner = "", []  # owner[k] = alignment entry that produced text[k]
        for i, piece in enumerate(pieces):
            text += piece
            owner.extend([i] * len(piece))
        words = [
            {
                "word": m.group(),
                "start": round(starts[owner[m.start()]], 3),
                "end": round(ends[owner[m.end() - 1]], 3),
            }
            for m in re.finditer(r"\S+", text)
        ]
        if not words:
            raise ValueError("ElevenLabs returned alignment without spoken words")
        return words
