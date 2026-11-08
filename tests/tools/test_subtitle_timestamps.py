"""Regression tests: SRT/VTT timestamp formatting must not overflow milliseconds.

`_ts_srt` / `_ts_vtt` computed the seconds and millisecond fields independently:
`ms = int(round((seconds % 1) * 1000))`. When the fractional part is >= 0.9995,
that rounds to 1000, emitting a malformed 4-digit `…,1000` with no carry into
the seconds (and, at boundaries, minutes/hours) field — e.g. 0.9999s ->
`00:00:00,1000` instead of `00:00:01,000`. Such timestamps are rejected by
strict SRT/VTT parsers (ffmpeg subtitles filter, VLC, browser WebVTT).
"""

import re
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.subtitle.subtitle_gen import SubtitleGen  # noqa: E402

_SRT_RE = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}$")
_VTT_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}$")


def test_millisecond_carry_does_not_overflow():
    # Every fractional part >= 0.9995 must carry into the next second, never
    # emit a 4-digit millisecond field.
    assert SubtitleGen._ts_srt(0.9999) == "00:00:01,000"
    assert SubtitleGen._ts_vtt(0.9999) == "00:00:01.000"
    assert SubtitleGen._ts_srt(1.9996) == "00:00:02,000"


def test_carry_propagates_across_minute_and_hour_boundaries():
    assert SubtitleGen._ts_srt(59.9999) == "00:01:00,000"
    assert SubtitleGen._ts_srt(3599.9999) == "01:00:00,000"
    assert SubtitleGen._ts_srt(7261.9999) == "02:01:02,000"


def test_normal_values_unchanged():
    assert SubtitleGen._ts_srt(0.0) == "00:00:00,000"
    assert SubtitleGen._ts_srt(1.5) == "00:00:01,500"
    assert SubtitleGen._ts_srt(0.4994) == "00:00:00,499"
    assert SubtitleGen._ts_vtt(83.25) == "00:01:23.250"


def test_all_outputs_are_well_formed():
    # Sweep values that land on and around the dangerous boundary.
    for t in (0.0, 0.4995, 0.9995, 0.9999, 1.0, 59.9995, 3599.9999, 12345.9999):
        assert _SRT_RE.match(SubtitleGen._ts_srt(t)), SubtitleGen._ts_srt(t)
        assert _VTT_RE.match(SubtitleGen._ts_vtt(t)), SubtitleGen._ts_vtt(t)


def test_strict_alignment_rejects_segment_only_timing(tmp_path):
    output = tmp_path / "captions.srt"
    result = SubtitleGen().execute({
        "segments": [{"text": "Hello world", "start": 0.0, "end": 1.0}],
        "require_word_timestamps": True,
        "audio_duration_seconds": 1.0,
        "output_path": str(output),
    })
    assert not result.success
    assert "refusing segment-level" in result.error
    assert not output.exists()


def test_strict_alignment_rejects_out_of_bounds_and_nonchronological_words(tmp_path):
    for words in (
        [{"word": "late", "start": 0.8, "end": 1.2}],
        [
            {"word": "first", "start": 0.5, "end": 0.7},
            {"word": "second", "start": 0.4, "end": 0.8},
        ],
    ):
        result = SubtitleGen().execute({
            "segments": [{"words": words}],
            "require_word_timestamps": True,
            "audio_duration_seconds": 1.0,
            "output_path": str(tmp_path / "captions.srt"),
        })
        assert not result.success
        assert "timestamp" in result.error.lower()


def test_strict_alignment_groups_at_sentence_and_preserves_remotion_word_times(tmp_path):
    words = [
        {"word": "Artificial", "start": 0.0, "end": 0.3},
        {"word": "intelligence", "start": 0.31, "end": 0.75},
        {"word": "helps", "start": 0.9, "end": 1.2},
        {"word": "people.", "start": 1.21, "end": 1.55},
        {"word": "Really?", "start": 1.8, "end": 2.2},
    ]
    srt = tmp_path / "captions.srt"
    result = SubtitleGen().execute({
        "segments": [{"words": words}],
        "require_word_timestamps": True,
        "audio_duration_seconds": 2.3,
        "format": "srt",
        "output_path": str(srt),
        "max_words_per_cue": 8,
    })
    assert result.success, result.error
    content = srt.read_text(encoding="utf-8")
    assert "Artificial intelligence helps people." in content
    assert "Really?" in content
    assert len(result.data["remotion_captions"]) == len(words)
    assert result.data["remotion_captions"][3] == {
        "word": "people.", "startMs": 1210, "endMs": 1550, "pageBreakAfter": True
    }


def test_json_sidecar_contains_remotion_caption_schema(tmp_path):
    output = tmp_path / "captions.caption.json"
    result = SubtitleGen().execute({
        "segments": [{"words": [
            {"word": "Hello", "start": 0.1, "end": 0.4},
            {"word": "world!", "start": 0.41, "end": 0.8},
        ]}],
        "require_word_timestamps": True,
        "audio_duration_seconds": 1.0,
        "format": "json",
        "output_path": str(output),
    })
    assert result.success, result.error
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["captions"] == result.data["remotion_captions"]
    assert payload["captions"][0] == {"word": "Hello", "startMs": 100, "endMs": 400}
