# Subtitle Sync Skill

## When to Use

Use `subtitle_gen` to format actual word-level timestamps as subtitles. For
new ElevenLabs narration, request `timestamps: true` from `tts_selector`; its
`/with-timestamps` result includes character alignment, converted word times,
and an `<audio>.alignment.json` sidecar. For an existing recording without a
sidecar, use the registered `elevenlabs_forced_alignment` tool with the exact
script and audio. That endpoint consumes Speech-to-Text minutes. The local
`transcriber` is an ASR tool that can timestamp its recognized words; it is
not forced alignment against a supplied script.

## Tool

| Tool | Capability |
|------|-----------|
| `subtitle_gen` | Generate SRT, VTT, or caption JSON from word-level timestamps |

## Output Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| SRT | `.srt` | Universal — works with FFmpeg, players, YouTube upload |
| VTT | `.vtt` | Web-native — HTML5 video, browser playback |
| Caption JSON | `.caption.json` | Programmatic — word-level data for custom renderers |

## Cue Length by Format

### Vertical Short-form (TikTok, Reels, Shorts)

- **Max 3-4 words per cue** — screen is narrow, text must be large enough to read
- **Max 20 characters per line** — prevents wrapping on narrow screens
- Subtitles are **mandatory** (most viewers watch muted)

### Horizontal Standard (YouTube, web)

- **Max 6-8 words per cue** — wider screen accommodates more text
- **Max 42 characters per line** — standard broadcast limit

### General Rules

- Average viewer reads ~15 characters/second
- Minimum display time: 0.5 seconds per cue
- Maximum display time: 5 seconds per cue

## Styling for Burn-in (ASS force_style)

When burning subtitles via `video_compose`, these parameters are passed as ASS
`force_style`. Use the correct ASS color format: `&HAABBGGRR` (not hex RGB).

### Vertical Video (1080x1920)

```
font: Arial
font_size: 18
bold: true
primary_color: &H00FFFFFF      # white (ASS format: alpha=00, BGR=FFFFFF)
outline_color: &H00000000      # black
outline_width: 3               # thick outline for readability on varied backgrounds
shadow: 2
margin_v: 50                   # pixels from bottom edge
alignment: 2                   # bottom center
```

### Horizontal Video (1920x1080)

```
font: Arial
font_size: 22
bold: true
primary_color: &H00FFFFFF
outline_color: &H00000000
outline_width: 2
shadow: 1
margin_v: 40
alignment: 2
```

### Common Mistakes

- **Wrong color format:** `&HFFFFFF` breaks positioning. Always use full 8-char `&H00FFFFFF`.
- **Font too large on vertical:** `font_size: 28` fills the center of a 9:16 frame. Use 18 max.
- **Too many words per cue on vertical:** 5+ words creates multi-line blocks that cover the face.
- **MarginV too large:** Values over 200 push text off-screen. Stay under 100 for most cases.

## Timing Best Practices

### Alignment with Speech

- Cue start must match word onset (not before the speaker starts)
- Cue end should extend ~200ms past the last word for comfortable reading
- Never let a cue linger into the next speaker's turn

### Word Boundary Grouping

The `subtitle_gen` tool groups words respecting `max_words_per_cue` and
`max_chars_per_line`, preserves punctuation, and ends cues at sentence
boundaries when practical. For production subtitles, pass segments containing
`words` and set `require_word_timestamps: true` plus
`audio_duration_seconds`. The strict mode rejects missing, invalid,
non-chronological, negative, or out-of-audio timings instead of emitting
segment-only cues. The default permissive mode remains for legacy workflows;
its segment-only times are coarse segment boundaries and must not be called
word alignment.

Example input shape:

```json
{
  "segments": [{"words": [
    {"word": "Hello,", "start": 0.12, "end": 0.48},
    {"word": "world!", "start": 0.51, "end": 0.91}
  ]}],
  "require_word_timestamps": true,
  "audio_duration_seconds": 1.0
}
```

SRT remains compatible with FFmpeg burn-in. `format: "json"` also emits a
`captions` array using Remotion's `{word, startMs, endMs}` word-caption
contract; pass that array as the composition `captions` prop. No estimated
fallback is allowed after a requested alignment operation fails: stop, report
the failure, and correct the alignment input or explicitly choose a separately
approved ASR/realignment path.

## Alignment quality report

`elevenlabs_forced_alignment` writes the parsed provider response (including
`raw_api_response`) plus a separate `.quality.json` report. The report includes
word coverage, timestamp errors, overall and maximum per-word loss, the ten
highest numeric-loss entries, and bottom/top duration-decile review candidates.
It never drops or retimes a questionable word. The relative loss list is not an
absolute confidence classification, and duration deciles are review heuristics.

ElevenLabs describes overall `loss` as the average alignment loss/confidence
score over characters and returns per-word loss values, but publishes no score
range, direction, or acceptance threshold. Read the largest values as
comparative review candidates only. The quality report therefore defaults to
`human_review_required` and `subtitles_fully_validated: false`; a complete word
match alone does not mark the audio reviewed. Missing words or invalid times
set `quality_status: "blocked"` and must stop subtitle production.

## Troubleshooting

- `elevenlabs_forced_alignment` status confirms only that the key is configured
  and `ffprobe` is present; it cannot check endpoint permissions offline.
- HTTP 401 mentioning the `forced_alignment` permission means the key is
  configured but lacks access to that endpoint. TTS access can still work.
  Do not retry or fall back to estimated cues; enable the permission through
  the provider's key settings, or explicitly choose another alignment method.
- A provider alignment error should be reported as a blocker. Strict
  `subtitle_gen` mode rejects absent, malformed, non-chronological, negative,
  or out-of-duration word timings.

## Quality Checklist

- [ ] Every spoken word appears in a subtitle cue
- [ ] No cue exceeds the character limit for the target format
- [ ] Subtitles are in the bottom 20% of frame — never covering the face
- [ ] Text is readable on mobile at native resolution
- [ ] Timing matches speech — no early or late cues
- [ ] Cues don't overlap each other
- [ ] Outline/shadow provides sufficient contrast against all backgrounds
