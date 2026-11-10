# Compose Director — Explainer Pipeline

## When to Use

You are the Compositor for a generated explainer video. You have `edit_decisions` with the complete edit timeline and an `asset_manifest` with all file paths. Your job is to render the final video: assemble visuals, layer audio, burn subtitles, and encode to the target format.

This is the last technical stage before the video exists as a playable file. Everything converges here.

## Runtime Routing (MANDATORY first step)

Read `edit_decisions.render_runtime` before anything else. It was locked at proposal and must not be changed silently. The rest of this skill's process steps (Remotion public/ staging, word-level caption burn, etc.) assume `render_runtime="remotion"` — the default for data-driven explainers.

- **`render_runtime="hyperframes"`** — HTML/CSS/GSAP render. Do NOT follow the Remotion-specific steps below. Instead: read `skills/core/hyperframes.md`, `.agents/skills/hyperframes/SKILL.md`, and `.agents/skills/hyperframes-cli/SKILL.md`. Call `video_compose` with the edit_decisions unchanged — it will delegate to `hyperframes_compose`, which materializes a workspace under `projects/<name>/hyperframes/`, runs `lint → validate → render`, and returns the MP4. Both lint AND validate must pass before render; contrast can be deferred during iteration but not for final delivery.
- **`render_runtime="ffmpeg"`** — simple concat/trim. Call `video_compose` directly; it will NOT auto-upgrade to Remotion when this runtime is explicitly locked.
- **Runtime unavailable** — surface the blocker per AGENT_GUIDE.md > "Escalate Blockers Explicitly" and get user approval (recorded as a `render_runtime_selection` decision in decision_log) before switching.

`final_review.checks.promise_preservation.render_runtime_used` must equal the runtime that actually ran; `runtime_swap_detected` must be `false` unless an approved decision authorizes the swap.

**Pass `proposal_packet` to `video_compose.execute()`** so in-tool swap detection can actually fire. Without it the `runtime_swap_check` is reported as `skipped` and you have to rely on the reviewer skill's cross-artifact comparison instead.

## Prerequisites

| Layer | Resource | Purpose |
|-------|----------|---------|
| Schema | `schemas/artifacts/render_report.schema.json` | Artifact validation |
| Prior artifacts | `state.artifacts["edit"]["edit_decisions"]`, `state.artifacts["assets"]["asset_manifest"]` | What to render |
| Playbook | Active style playbook | Quality targets |
| Tools | `video_compose`, `audio_mixer` | Rendering capabilities |
| Media profiles | `lib/media_profiles.py` | Output format specs (resolution, codec, bitrate) |

## Process

### Step 1: Choose Render Strategy

Before rendering an explainer, read the stage `composition_plan` and its latest
`composition_quality` report. Apply its per-cut crop mode/focal point,
`motion_intensity`, scene graphics, and transition through the existing
Remotion `Explainer` cuts contract. `cover` uses the configured focal point;
`contain_blur` preserves all source content against a softened backplate.
Graphics are scene-local and must remain clear of the plan's subtitle safe area.
The renderer does not infer subject position, judge relevance, or resolve
warnings. Stop on technical errors; surface unresolved visual warnings and
require human review before a production render. Keep caption timing unchanged.

Still-image parallax is not currently true depth-aware parallax; use restrained
frame-driven pan/zoom and state that limitation in the plan.

Based on the edit decisions, pick the rendering approach:

**Remotion render** (DEFAULT — use this unless explicitly overridden):
- Animated text cards, stat cards, chart scenes
- Complex transitions (morph, zoom, ken-burns)
- Programmatic motion graphics
- Audio embedding (narration + music with fade/volume)
- Word-level captions via CaptionOverlay component
- Best for: ALL explainer videos, both image-based and animation-heavy

**FFmpeg pipeline** (FALLBACK — only when Remotion is unavailable):
- Static images with Ken Burns
- Audio layering
- SRT subtitle burn-in
- Best for: environments without Node.js/Remotion installed

**IMPORTANT: When using Remotion, ALL of these go through Remotion — not FFmpeg:**
- Audio (narration + music) → Remotion `audio` prop, NOT external audio_mixer
- Subtitles → Remotion `captions` prop (word-level), NOT SRT burn via FFmpeg
- Text overlays (CTA, titles) → Remotion `text_card` cut type, NOT AI-generated images

### Step 2: Audio Acquisition (Narration, Music, Subtitles)

Before rendering, present the user with audio options and get their preferences.

**Present to the user:**

> **Audio setup for this video:**
>
> **Narration:** I can generate TTS narration using OpenAI TTS (`gpt-4o-mini-tts` — $0.015/min, 6 voices, voice direction). Which voice and tone would you like? I'll propose a voice based on the video topic, or you can choose:
> - `onyx` — deep, authoritative (documentaries, tech)
> - `echo` — resonant, futuristic (product ads, sci-fi)
> - `nova` — bright, energetic (upbeat, explainers)
> - `fable` — warm, storytelling (narratives, education)
> - `shimmer` — expressive, warm (organic, lifestyle)
> - `alloy` — neutral, balanced (general purpose)
>
> **Music:** I can automatically find royalty-free background music from Pixabay (no key needed). If you have a `FREESOUND_API_KEY`, I can also search Freesound as a backup.
>
> **Subtitles:** New ElevenLabs narration can return aligned word timings with the TTS request. Existing narration without that sidecar can be aligned to the approved script with ElevenLabs Forced Alignment; this uses Speech-to-Text minutes. The local transcriber is an ASR alternative, not forced alignment to the supplied script.
>
> Want me to proceed with my recommendations, or adjust anything?

**After user confirms:**

1. **Write narration script with duration budget** (see scene-director Step 4b):
   - Calculate video duration from cuts
   - Budget at 85-90% of video duration
   - Use 2.0-2.5 words/sec for documentary, 2.5-3.0 for energetic
   - Verify word count before generating TTS

2. **Generate TTS narration through `tts_selector`:**
   ```python
   from tools.audio.tts_selector import TTSSelector
   result = TTSSelector().execute({
       'text': narration_script,
       'timestamps': True,
       'output_path': 'path/to/narration.mp3',
   })
   # Preserve result.data['alignment_path'] and result.data['word_timestamps'].
   # Confirm alignment_matches_text before using script-exact captions.
   # CRITICAL: Check result.data['audio_duration_seconds'] vs video duration
   # If narration exceeds video by >1s: shorten script and regenerate
   ```

3. **Download background music:**
   ```python
   from tools.audio.pixabay_music import PixabayMusic
   result = PixabayMusic().execute({
       'query': '<mood/genre matching video topic>',
       'min_duration': video_duration_seconds,
       'max_duration': 300,
       'output_path': 'path/to/music.mp3',
   })
   ```

4. **Use returned TTS alignment, or forced-align an existing narration file:**
   ```python
   from tools.analysis.elevenlabs_forced_alignment import ElevenLabsForcedAlignment
   result = ElevenLabsForcedAlignment().execute({
       'audio_path': 'path/to/narration.wav',
       'text': approved_narration_script,
       'output_path': 'projects/<project>/artifacts/narration.alignment.json',
   })
   # This bills at the account's Speech-to-Text rate. The tool validates
   # chronological word timings against ffprobe duration and writes the raw
   # provider payload plus a separate quality report.
   ```

   Read `result.data["quality_report"]` (or the recorded quality-report path)
   before building captions. If its `quality_status` is `blocked`, stop and do
   not create subtitles. For `human_review_required`, present the ranked loss
   candidates and duration candidates for review; do not claim the subtitles
   are fully validated until a human checks them. ElevenLabs documents no
   absolute loss threshold, so the report ranks values without inventing one.

   Convert validated words to SubtitleGen segments (`[{"words": word_timestamps}]`),
   then set `require_word_timestamps: true` and pass the probed audio duration.
   For SRT use the returned file for FFmpeg. For Remotion use
   `SubtitleGen` `format: "json"` and pass `data.captions` as composition props;
   it emits `{word, startMs, endMs}` values compatible with `CaptionOverlay`.
   If TTS timestamps or forced alignment are missing/invalid, stop and report the
   blocker. Do not use estimated segment timing. ASR via `transcriber` may only
   be selected explicitly and its transcript must be checked against the script.

5. **Assemble composition JSON** with audio config:
   ```json
   {
     "audio": {
       "narration": { "src": "path/to/narration.mp3", "volume": 1 },
       "music": { "src": "path/to/music.mp3", "volume": 0.1, "fadeInSeconds": 2, "fadeOutSeconds": 3 }
     },
     "captions": [ ... validated word-level captions from SubtitleGen ... ]
   }
   ```

### Step 3: Prepare Render Inputs

For sourced footage, use only the approved scene selections recorded in
`footage_selection_plan` and the matching project-local files in
`asset_manifest`. The render contract remains the existing Remotion
composition: edit cuts reference manifest asset IDs/paths, while provider,
licence, and selection rationale remain provenance metadata. Do not search for
or substitute different footage during composition; return to the assets stage
if a chosen clip is missing or unsuitable.

For each cut in the edit decisions:
1. Verify the source asset exists at its declared path
2. Check asset dimensions/duration match expectations
3. Prepare transform parameters (scale, position, crop)

For audio:
1. Verify narration duration fits within video duration (use `audio_probe`)
2. Verify music duration covers video duration
3. Prepare ducking parameters from edit decisions

### Step 3: Determine Output Profile

Read the target platform from the brief artifact. Map to a media profile:

| Platform | Profile | Resolution | Notes |
|----------|---------|-----------|-------|
| YouTube | `youtube_landscape` | 1920x1080 | Default for most explainers |
| TikTok/Reels | `tiktok` | 1080x1920 | Vertical, needs reframing |
| Twitter/X | `twitter_landscape` | 1280x720 | Shorter format |
| LinkedIn | `linkedin` | 1920x1080 | Professional context |

Get the exact encoding parameters via `ffmpeg_output_args(get_profile(name))`.

### Step 4: Render Video

Call the `video_compose` tool with:
```
{
  "operation": "render",
  "edit_decisions": <edit_decisions artifact>,
  "asset_manifest": <asset_manifest artifact>,
  "output_profile": "youtube_landscape",
  "output_path": "renders/output.mp4",
  "options": {
    "subtitle_burn": true,
    "audio_normalize": true,
    "two_pass_encode": true
  }
}
```

If using Remotion for animated segments:
1. Generate Remotion composition data from edit decisions
2. Call `video_compose` with `operation: "remotion_render"` for animated segments
3. Assemble Remotion outputs with remaining segments via FFmpeg

**Zero-key Remotion render (component-only videos):**
When all scenes are Remotion component types (hero_title, stat_card, bar_chart, line_chart,
pie_chart, kpi_grid, comparison, callout, progress_bar, text_card), render the entire video
as a single Remotion composition using the Explainer entry point. No FFmpeg assembly needed.
The edit_decisions cuts array maps directly to Remotion props. See `skills/core/remotion.md`
for the proven formula — especially the all-dark-background rule for visual consistency.

### Step 5: Audio Post-Processing

**Remotion path (DEFAULT):** Skip external audio mixing entirely. Remotion handles all audio
natively via `<Audio>` components. Pass audio sources in the composition props:
```json
{
  "audio": {
    "narration": { "src": "project/narration.mp3", "volume": 1.0 },
    "music": { "src": "project/music.mp3", "volume": 0.12, "fadeInSeconds": 1.5, "fadeOutSeconds": 2.5 }
  }
}
```
Remotion renders audio and video in a single pass — no external muxing needed.
Do NOT use `audio_mixer` for ducking/mixing when rendering via Remotion.

**FFmpeg fallback (ONLY when Remotion is unavailable):**
Call the `audio_mixer` tool to:
1. Layer narration segments in order
2. Mix background music at playbook volume
3. Apply ducking (music dips during narration)
4. Normalize overall audio levels
5. Output the final mixed audio track
The video_compose tool will mux this with the video.

### Step 5b: Generate Subtitles (Mandatory)

Subtitles are mandatory for all explainer content. Generate them from the narration audio — do NOT skip this step.

**Remotion path (DEFAULT — when using Remotion render):**

1. **Transcribe** the full narration using the `transcriber` tool (whisperx):
   ```python
   from tools.analysis.transcriber import Transcriber
   result = Transcriber().execute({
       'input_path': 'projects/<project>/assets/audio/narration_full.mp3',
       'model_size': 'base',
       'language': 'en',
       'output_dir': 'projects/<project>/assets/audio'
   })
   # result.data contains segments with word-level timestamps
   ```

2. **Convert to Remotion WordCaption format** (NOT SRT):
   ```python
   captions = []
   for segment in result.data['segments']:
       for word_info in segment.get('words', []):
           captions.append({
               'word': word_info['word'],
               'startMs': int(word_info['start'] * 1000),
               'endMs': int(word_info['end'] * 1000),
           })
   ```

3. **Add captions to composition props** — they go in the `captions` array alongside `cuts` and `audio`:
   ```json
   {
     "cuts": [...],
     "audio": {...},
     "captions": [
       { "word": "Root", "startMs": 120, "endMs": 340 },
       { "word": "canals", "startMs": 340, "endMs": 680 }
     ]
   }
   ```

   Remotion's CaptionOverlay renders these as word-by-word highlighted captions with the theme's
   `captionHighlightColor` and `captionBackgroundColor`. This is superior to FFmpeg SRT burn because
   it produces animated word-level highlighting synchronized to narration.

**FFmpeg fallback (ONLY when Remotion is unavailable):**

If Remotion is not available, fall back to SRT generation + FFmpeg burn:
   ```python
   from tools.subtitle.subtitle_gen import SubtitleGen
   SubtitleGen().execute({
       'segments': [{'words': aligned_word_timestamps}],
       'require_word_timestamps': True,
       'audio_duration_seconds': narration_duration_seconds,
       'format': 'srt',
       'output_path': 'projects/<project>/assets/subtitles.srt',
       'max_words_per_cue': 8,
       'max_chars_per_line': 42
   })
   # Then burn with video_compose operation='burn_subtitles'
   ```

**The final deliverable MUST have subtitles** — either via Remotion captions or FFmpeg burn.

### Step 5c: Pre-Render Validation (Mandatory)

**Always run the composition validator before rendering.** This catches problems that waste render time.

```python
from tools.analysis.composition_validator import CompositionValidator
result = CompositionValidator().execute({
    'composition_path': 'path/to/composition.json',
    'assets_root': 'remotion-composer/public',
})
# result.data['valid'] MUST be True before proceeding to render
# If False: fix the reported errors first (missing assets, audio-video mismatch, etc.)
```

Common catches:
- Narration audio longer than video (would be cut off)
- Missing image/audio files (render would fail)
- Music shorter than video (silence at end)

**Do not skip this step.** If validation fails, fix the issue and re-validate before rendering.

### Step 6: Post-Render Self-Review (Mandatory — ALL steps required)

After rendering, the agent **must review its own output** before presenting to the user. This catches issues the validator can't see (visual quality, audio sync, subtitle readability).

**CRITICAL: You MUST complete ALL of steps 6a through 6e. Do NOT skip any step.
The most common agent failure is doing 6a (frames) and 6c (visual) while skipping
6b (audio transcription) — which misses catastrophic issues like missing audio entirely.**

**6a. Probe rendered file (FIRST — gate for all other checks):**
```bash
ffprobe -v quiet -print_format json -show_format -show_streams rendered_video.mp4
```
Verify:
- Video stream exists (codec_type: "video") with correct resolution
- **Audio stream exists (codec_type: "audio")** — if NO audio stream, STOP and fix immediately
- Duration is within ±5% of target
- File size is reasonable (not 0 bytes)

**If audio stream is missing: the render did not embed audio. Do NOT proceed to present
the video to the user. Fix the audio configuration and re-render.**

**6b. Extract review frames:**
```python
from tools.analysis.frame_sampler import FrameSampler
midpoints = [(cut['in_seconds'] + cut['out_seconds']) / 2 for cut in cuts]
FrameSampler().execute({
    'input_path': 'path/to/rendered_video.mp4',
    'strategy': 'timestamps',
    'timestamps': midpoints,
    'output_dir': 'path/to/review-frames',
    'format': 'png',
})
```

**6c. Transcribe rendered audio (MANDATORY — do NOT skip):**
```python
from tools.analysis.transcriber import Transcriber
result = Transcriber().execute({
    'input_path': 'path/to/rendered_video.mp4',
    'model_size': 'base',
    'language': 'en',
    'output_dir': 'path/to/review-frames',
})
# If result returns 0 words: audio is silent/missing — STOP and fix
# If word count < 80% of script word count: audio is cut off — investigate
```

**6d. Visual inspection — review each frame:**
- Does the background color/gradient match intent? (watch for white backgrounds on dark-themed videos)
- Are images rendering correctly? (not blank, not stretched)
- Are subtitles/captions visible and properly spaced?
- Are overlays (section titles, stat reveals) positioned correctly?
- Is the opening scene visually strong? (important for social media thumbnails)
- Does the CTA/closing screen show correct text? (AI-generated text in images frequently hallucinates — use Remotion text_card for any text that must be exact)

**6e. Audio inspection — check transcript against script:**
- Is the full narration captured? (compare last transcribed word to last scripted word)
- Any words cut off at the end? (narration exceeding video duration)
- Timing alignment — do narration segments roughly match their intended scenes?
- Is background music audible? (transcriber may not capture music, but ffprobe confirms audio stream)

**6f. Compile and present review to user:**

> **Post-render review for "[Video Title]":**
>
> **File:** [duration]s, [resolution], [file size] — audio stream: [present/MISSING]
> **Audio:** [Complete/Cut off at Xs] — [N]/[M] words transcribed from rendered output
> **Visuals:** [N scenes inspected] — [issues or "all scenes rendering correctly"]
> **Captions:** [Remotion CaptionOverlay / FFmpeg SRT / MISSING] — [word-level highlight working / issues]
> **Issues found:** [list any issues with severity]
>
> **Recommendations:** [what to fix, if anything]
>
> Want me to fix these issues and re-render, or is this good to go?

**Only after user approves (or agent finds zero issues) should the video be considered final.**

### Step 6-old: File and Content Verification

**File verification:**
- [ ] Output file exists at declared path
- [ ] File size is reasonable (not 0 bytes, not suspiciously small)
- [ ] File is a valid container (ffprobe succeeds)

**Content verification:**
- [ ] Duration within ±5% of target
- [ ] Resolution matches selected profile
- [ ] Audio channels present (stereo)
- [ ] No audio clipping or silence gaps > 1s

**Quality check (covered by self-review above):**
- [ ] Visual: all scene frames inspected
- [ ] Audio: full transcription verified
- [ ] Subtitles: visible and correctly timed

### Step 7: Build Render Report

```json
{
  "version": "1.0",
  "outputs": [
    {
      "path": "renders/output.mp4",
      "format": "mp4",
      "codec": "h264",
      "resolution": "1920x1080",
      "fps": 30,
      "duration_seconds": 62.4,
      "file_size_mb": 45.2,
      "audio_codec": "aac",
      "audio_channels": 2,
      "render_strategy": "ffmpeg",
      "render_time_seconds": 180
    }
  ],
  "render_summary": {
    "total_cuts_rendered": 12,
    "subtitles_burned": true,
    "audio_tracks_mixed": 3,
    "target_duration_seconds": 60,
    "actual_duration_seconds": 62.4
  }
}
```

### Step 8: Self-Evaluate

Score (1-5):

| Criterion | Question |
|-----------|----------|
| **Playability** | Does the video play without errors in a standard player? |
| **Duration accuracy** | Is actual duration within ±5% of target? |
| **Audio quality** | Is narration clear, music balanced, no clipping? |
| **Visual quality** | Are images sharp, transitions smooth, no artifacts? |
| **Subtitle accuracy** | Are subtitles present, readable, and synced? |

If any dimension scores below 3, investigate and re-render.

### Step 9: Submit

Validate the render_report against the schema and persist via checkpoint.

## Common Pitfalls

- **Missing asset files**: Always verify every referenced file exists before starting the render. A missing file mid-render wastes time.
- **Audio sync drift**: Accumulated timing errors across narration segments cause audio-visual desync. Use absolute timestamps, not relative offsets.
- **Subtitle encoding**: Burn subtitles into the video (hardcoded) for maximum compatibility. Don't rely on soft subtitles for social media.
- **Single-pass encode**: Two-pass encoding produces better quality at the same file size. Worth the extra render time.
- **Ignoring media profile**: YouTube and TikTok have very different requirements. Always check the target profile.
