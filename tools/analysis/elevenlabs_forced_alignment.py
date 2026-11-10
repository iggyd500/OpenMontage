"""Align a supplied transcript to an existing audio file with ElevenLabs.

This complements the ElevenLabs TTS ``with_timestamps`` option: use that
option when synthesizing new narration, and this tool only when an existing
recording has no alignment sidecar.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests

from tools.analysis.audio_probe import probe_duration
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
from tools.provider_pricing import PriceQuoteRequired


class ElevenLabsForcedAlignment(BaseTool):
    name = "elevenlabs_forced_alignment"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "speech_alignment"
    provider = "elevenlabs"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.API

    dependencies = ["env:ELEVENLABS_API_KEY", "binary:ffprobe"]
    install_instructions = (
        "Configure ELEVENLABS_API_KEY and ensure ffprobe is on PATH."
    )
    capabilities = ["forced_alignment", "word_timestamps", "character_timestamps"]
    supports = {
        "existing_audio": True,
        "word_timestamps": True,
        "character_timestamps": True,
        "confidence_loss": True,
        "gpu_required": False,
    }
    best_for = [
        "aligning a supplied transcript to existing narration audio",
        "word-level subtitle timing when TTS alignment was not saved",
    ]
    retry_policy = RetryPolicy(max_retries=0)
    idempotency_key_fields = ["audio_path", "text"]
    side_effects = ["calls ElevenLabs Forced Alignment API", "writes alignment JSON"]
    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=128, vram_mb=0, disk_mb=10, network_required=True
    )

    input_schema = {
        "type": "object",
        "required": ["audio_path", "text", "output_path"],
        "properties": {
            "audio_path": {"type": "string"},
            "text": {"type": "string", "minLength": 1},
            "output_path": {"type": "string"},
            "quality_report_path": {
                "type": "string",
                "description": "Optional path for a separate alignment-quality JSON report.",
            },
            "timeout_seconds": {"type": "number", "default": 120, "minimum": 1},
        },
    }

    def get_status(self) -> ToolStatus:
        return (
            ToolStatus.AVAILABLE
            if os.environ.get("ELEVENLABS_API_KEY")
            and shutil.which("ffprobe")
            else ToolStatus.UNAVAILABLE
        )

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Account tier and included STT minutes vary; never present a guessed
        # dollar figure as the account's actual charge.
        raise PriceQuoteRequired(
            "ElevenLabs Forced Alignment is billed at the account's Speech-to-Text rate; verify plan-specific pricing."
        )

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        audio_path = Path(inputs["audio_path"])
        text = inputs["text"]
        output_path = Path(inputs["output_path"])
        quality_path = Path(
            inputs.get("quality_report_path")
            or output_path.with_name(f"{output_path.stem}.quality.json")
        )
        api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        if not api_key:
            return ToolResult(success=False, error="ELEVENLABS_API_KEY is not configured")
        if not audio_path.is_file():
            return ToolResult(success=False, error=f"Audio file not found: {audio_path}")
        if not text.strip():
            return ToolResult(success=False, error="Transcript text must not be empty")
        if output_path.exists():
            return ToolResult(success=False, error=f"Refusing to overwrite alignment file: {output_path}")
        if quality_path == output_path or quality_path.exists():
            return ToolResult(success=False, error=f"Refusing to overwrite or alias quality report: {quality_path}")
        audio_duration = probe_duration(audio_path)
        if audio_duration is None or audio_duration <= 0:
            return ToolResult(success=False, error="Could not determine source audio duration with ffprobe")

        started = time.perf_counter()
        try:
            with audio_path.open("rb") as audio_file:
                response = requests.post(
                    "https://api.elevenlabs.io/v1/forced-alignment",
                    headers={"xi-api-key": api_key},
                    files={"file": (audio_path.name, audio_file)},
                    data={"text": text},
                    timeout=float(inputs.get("timeout_seconds", 120)),
                )
            response.raise_for_status()
            payload = response.json()
            quality = self.build_quality_report(
                text, payload.get("words"), payload.get("loss"), audio_duration
            )
            validation_error = None
            words = []
            characters = []
            try:
                words = self._validate_words(payload.get("words"), audio_duration)
                characters = self._validate_characters(payload.get("characters"), audio_duration)
                matches_script = self._normalized_words(text) == self._normalized_words(
                    " ".join(word["word"] for word in words)
                )
                if not matches_script:
                    raise ValueError(
                        "word alignment does not match the supplied transcript; refusing subtitle timing"
                    )
            except (ValueError, TypeError, KeyError) as exc:
                validation_error = str(exc)
                quality["quality_status"] = "blocked"
                quality["warnings"].append(validation_error)
                quality["subtitles_fully_validated"] = False

            elapsed = round(time.perf_counter() - started, 3)
            quality["raw_api_response_preserved"] = True
            result = {
                "provider": self.provider,
                "method": "elevenlabs_forced_alignment",
                "audio_path": str(audio_path),
                "audio_duration_seconds": round(audio_duration, 3),
                "text": text,
                "alignment_matches_text": not quality["missing_words"]
                and not quality["unmatched_words"],
                "word_timestamps": words,
                "character_timestamps": characters,
                "alignment_loss": payload.get("loss"),
                "raw_api_response": payload,
                "quality_report": quality,
                "quality_report_path": str(quality_path),
                "generation_time_seconds": elapsed,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
            if validation_error:
                return ToolResult(
                    success=False,
                    data={"quality_report": quality, "quality_report_path": str(quality_path)},
                    artifacts=[str(output_path), str(quality_path)],
                    error=f"Alignment quality blocked subtitle timing: {validation_error}",
                    cost_usd=None,
                    duration_seconds=elapsed,
                )
            return ToolResult(
                success=True,
                data={**result, "output": str(output_path)},
                artifacts=[str(output_path), str(quality_path)],
                cost_usd=None,
                duration_seconds=elapsed,
            )
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            detail = self._safe_error_detail(exc.response, api_key)
            return ToolResult(
                success=False,
                error=f"ElevenLabs forced alignment failed (HTTP {status}){': ' + detail if detail else ''}",
                cost_usd=None,
                duration_seconds=round(time.perf_counter() - started, 3),
            )
        except requests.RequestException as exc:
            message = str(exc).replace(api_key, "[REDACTED]")
            return ToolResult(
                success=False,
                error=f"ElevenLabs forced alignment request failed: {message[:500]}",
                cost_usd=None,
                duration_seconds=round(time.perf_counter() - started, 3),
            )
        except (ValueError, KeyError, TypeError) as exc:
            message = str(exc).replace(api_key, "[REDACTED]")
            return ToolResult(
                success=False,
                error=f"Invalid forced-alignment response: {message[:500]}",
                duration_seconds=round(time.perf_counter() - started, 3),
            )

    @staticmethod
    def _safe_error_detail(response: requests.Response | None, api_key: str) -> str:
        if response is None:
            return ""
        try:
            body = response.json()
        except ValueError:
            return ""
        if not isinstance(body, dict):
            return ""
        detail = body.get("detail", body.get("message", ""))
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("status") or detail.get("type") or ""
        return str(detail).replace(api_key, "[REDACTED]")[:400]

    @classmethod
    def build_quality_report(
        cls,
        text: str,
        provider_words: Any,
        overall_loss: Any,
        audio_duration: float,
    ) -> dict[str, Any]:
        """Summarize alignment health without inventing provider score cutoffs.

        Loss candidates are ranked relative to this response. Duration candidates
        are the bottom/top decile of this response's valid word durations; both
        lists are review heuristics, not confidence judgments.
        """
        entries = provider_words if isinstance(provider_words, list) else []
        nonspace: list[tuple[int, str, Any, Any, Any]] = []
        invalid: list[dict[str, Any]] = []
        previous_start = previous_end = -1.0
        for index, item in enumerate(entries):
            if not isinstance(item, dict):
                invalid.append({"index": index, "word": None, "reasons": ["malformed entry"]})
                continue
            word = item.get("text", item.get("word"))
            if not isinstance(word, str):
                invalid.append({"index": index, "word": word, "reasons": ["missing word text"]})
                continue
            if not word.strip():
                continue
            start, end = item.get("start"), item.get("end")
            reasons = []
            numeric = isinstance(start, (int, float)) and isinstance(end, (int, float))
            finite = numeric and math.isfinite(start) and math.isfinite(end)
            if not numeric or not finite:
                reasons.append("timestamp is missing, non-numeric, or non-finite")
            else:
                if start < 0:
                    reasons.append("negative start")
                if end <= start:
                    reasons.append("end is not after start")
                if end > audio_duration + 0.05:
                    reasons.append("end exceeds audio duration")
                if start < previous_start or end < previous_end:
                    reasons.append("timestamps are non-chronological")
                previous_start, previous_end = start, end
            if reasons:
                invalid.append({"index": index, "word": word, "start": start, "end": end, "reasons": reasons})
            loss = item.get("loss")
            if isinstance(loss, (int, float)) and not math.isfinite(loss):
                invalid.append({"index": index, "word": word, "loss": loss, "reasons": ["loss is non-finite"]})
                loss = None
            nonspace.append((index, word, start, end, loss))

        expected_tokens = cls._normalized_words(text)
        aligned_text = " ".join(word for _, word, _, _, _ in nonspace)
        aligned_tokens = cls._normalized_words(aligned_text)
        missing: list[str] = []
        unmatched: list[str] = []
        for operation, a0, a1, b0, b1 in SequenceMatcher(
            None, expected_tokens, aligned_tokens, autojunk=False
        ).get_opcodes():
            if operation in ("delete", "replace"):
                missing.extend(expected_tokens[a0:a1])
            if operation in ("insert", "replace"):
                unmatched.extend(aligned_tokens[b0:b1])

        valid_entries = []
        for index, word, start, end, loss in nonspace:
            if (
                isinstance(start, (int, float)) and math.isfinite(start)
                and isinstance(end, (int, float)) and math.isfinite(end)
                and start >= 0 and end > start and end <= audio_duration + 0.05
            ):
                valid_entries.append({
                    "index": index,
                    "word": word,
                    "start": round(float(start), 3),
                    "end": round(float(end), 3),
                    "duration_seconds": round(float(end - start), 3),
                    "_duration_exact": float(end - start),
                    "loss": loss if isinstance(loss, (int, float)) else None,
                })

        loss_entries = [item for item in valid_entries if item["loss"] is not None]
        highest_loss = [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in sorted(loss_entries, key=lambda item: item["loss"], reverse=True)[:10]
        ]
        for rank, item in enumerate(highest_loss, 1):
            item["relative_loss_rank"] = rank

        durations = sorted(item["_duration_exact"] for item in valid_entries)

        def percentile(values: list[float], fraction: float) -> float | None:
            if not values:
                return None
            position = (len(values) - 1) * fraction
            lower, upper = math.floor(position), math.ceil(position)
            return values[lower] + (values[upper] - values[lower]) * (position - lower)

        short_cutoff = percentile(durations, 0.10)
        long_cutoff = percentile(durations, 0.90)
        short_words = [
            {key: value for key, value in x.items() if not key.startswith("_")}
            for x in valid_entries
            if short_cutoff is not None and x["_duration_exact"] <= short_cutoff + 1e-12
        ]
        long_words = [
            {key: value for key, value in x.items() if not key.startswith("_")}
            for x in valid_entries
            if long_cutoff is not None and x["_duration_exact"] >= long_cutoff - 1e-12
        ]
        if isinstance(overall_loss, (int, float)) and math.isfinite(overall_loss):
            reported_overall_loss = float(overall_loss)
        else:
            reported_overall_loss = None

        blocked = bool(missing or unmatched or invalid)
        warnings = [
            "ElevenLabs documents loss as alignment loss/confidence but specifies no score range, direction, or recommended threshold.",
            "Highest-loss entries are relative review candidates only; no provider-defined loss cutoff was applied.",
            "Duration candidates are the bottom/top 10th-percentile review sets for this alignment, not calibrated error thresholds.",
            "Human review is required before describing subtitles as fully validated.",
        ]
        if missing or unmatched:
            warnings.append("Aligned token sequence differs from the supplied transcript.")
        if invalid:
            warnings.append("One or more alignment entries have invalid timestamps or loss values.")

        return {
            "version": "1.0",
            "provider": "elevenlabs",
            "method": "elevenlabs_forced_alignment",
            "total_transcript_words": len(text.split()),
            "transcript_comparison_tokens": len(expected_tokens),
            "successfully_aligned_words": len(valid_entries),
            "aligned_comparison_tokens": len(aligned_tokens),
            "missing_words": missing,
            "unmatched_words": unmatched,
            "invalid_timestamp_words": invalid,
            "overall_alignment_loss": reported_overall_loss,
            "maximum_alignment_loss": max((item["loss"] for item in loss_entries), default=None),
            "highest_loss_review_candidates": highest_loss,
            "loss_review_policy": "Top 10 by numeric loss within this response; relative ranking only, not a confidence threshold.",
            "duration_review": {
                "method": "Distribution-based bottom/top decile candidate lists.",
                "short_duration_cutoff_seconds": round(short_cutoff, 3) if short_cutoff is not None else None,
                "short_duration_candidates": short_words,
                "long_duration_cutoff_seconds": round(long_cutoff, 3) if long_cutoff is not None else None,
                "long_duration_candidates": long_words,
                "thresholds_are_provider_defined": False,
            },
            "quality_status": "blocked" if blocked else "human_review_required",
            "review_required": True,
            "subtitles_fully_validated": False,
            "raw_api_response_preserved": False,
            "warnings": warnings,
            "audio_duration_seconds": round(audio_duration, 3),
        }

    @staticmethod
    def _validate_words(words: Any, duration: float) -> list[dict[str, Any]]:
        if not isinstance(words, list) or not words:
            raise ValueError("response has no word-level alignment")
        result = []
        previous_start = previous_end = -1.0
        for item in words:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError("response contains malformed word alignment")
            is_separator = not item["text"].strip()
            start, end = item.get("start"), item.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise ValueError("response contains non-numeric word timings")
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError("response contains non-finite word timings")
            if (
                start < 0 or end < start or (not is_separator and end <= start)
                or start < previous_start
                or end < previous_end or end > duration + 0.05
            ):
                raise ValueError("response word timings are non-chronological or outside audio duration")
            normalized = {
                "word": item["text"],
                "start": round(float(start), 3),
                "end": round(float(end), 3),
            }
            if "loss" in item and isinstance(item["loss"], (int, float)):
                if not math.isfinite(item["loss"]):
                    raise ValueError("response contains non-finite alignment confidence loss")
                normalized["loss"] = item["loss"]
            previous_start, previous_end = start, end
            # ElevenLabs can return whitespace separators in its `words` list.
            # They carry timing but are not spoken words and must not become
            # subtitle tokens. Keep only the actual lexical entries here.
            if not is_separator:
                result.append(normalized)
        if not result:
            raise ValueError("response has no spoken word-level alignment")
        return result

    @staticmethod
    def _validate_characters(characters: Any, duration: float) -> list[dict[str, Any]]:
        if not isinstance(characters, list):
            raise ValueError("response has malformed character alignment")
        result = []
        previous_start = previous_end = -1.0
        for item in characters:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError("response contains malformed character alignment")
            start, end = item.get("start"), item.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise ValueError("response contains non-numeric character timings")
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError("response contains non-finite character timings")
            if (
                start < 0 or end < start or start < previous_start
                or end < previous_end or end > duration + 0.05
            ):
                raise ValueError("response character timings are non-chronological or outside audio duration")
            result.append({"text": item["text"], "start": round(float(start), 3), "end": round(float(end), 3)})
            previous_start, previous_end = start, end
        return result

    @staticmethod
    def _normalized_words(text: str) -> list[str]:
        """Compare script and alignment words while ignoring punctuation/case."""
        return [word.casefold() for word in re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text)]
