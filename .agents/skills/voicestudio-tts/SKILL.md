---
name: voicestudio-tts
description: Generate local English narration through OpenMontage's VoiceStudio KittenTTS adapter. Use only when the user selects VoiceStudio or Jasper.
---

# VoiceStudio TTS for OpenMontage

Use this skill when a user explicitly chooses the local VoiceStudio TTS provider or the Jasper KittenTTS voice. OpenMontage routes speech through the existing `tts_selector`; invoke it with `preferred_tool: "voicestudio_tts"` and preserve the approved voice, speed, and model for every narration segment.

## Supported setup

- VoiceStudio backend must already be running at the configured local endpoint (default `http://localhost:3900`).
- KittenTTS Mini 0.8 must already be cached in VoiceStudio. OpenMontage checks the cache and refuses generation if its status is missing; it never launches VoiceStudio or downloads weights.
- Model ID is `kittentts`.
- Jasper is `expr-voice-2-m` (confirmed VoiceStudio KittenTTS preset); do not substitute another preset ID.
- English output is WAV. Other KittenTTS voices include Bella (`expr-voice-2-f`), Luna (`expr-voice-3-f`), Bruno (`expr-voice-3-m`), Rosie (`expr-voice-4-f`), Hugo (`expr-voice-4-m`), Kiki (`expr-voice-5-f`), and Leo (`expr-voice-5-m`).

## Production workflow

1. Produce a short approval sample using the exact selected preset and normal speed unless the user specifies otherwise.
2. Let the user listen and approve voice character, pronunciation, pacing, and pauses before producing the full narration.
3. Keep calls sequential. The adapter serializes local requests to reduce memory pressure; do not parallelize VoiceStudio sections.
4. Set a project-specific `output_path` for each section. Confirm returned WAV metadata and duration, then listen to the result before accepting it.
5. If the backend is stopped, the cache is absent, a request times out, or generation fails, report the adapter error and stop that route. Do not silently switch providers, start the app, or download a model. The user can explicitly choose Piper, ElevenLabs, or Google TTS as alternatives.

For quick social clips, Jasper is a reasonable quality-first local choice when the user accepts slower rendering. The evaluated Jasper run rendered at about 3× real time; for longer narration or faster iteration, prefer the configured Piper model unless the user approves the additional VoiceStudio render time.

VoiceStudio/KittenTTS is English-only in this adapter and does not provide word alignment through this integration. Build subtitles from the approved script or a separate transcription/alignment tool.
