# Intelligent Scene Composition (Stage 3C)

The animated-explainer edit stage can produce a `composition_plan` alongside
the existing `edit_decisions`. The plan keeps the approved Stage 3B assets and
adds per-scene narration, crop/focal point, protected source regions, still
motion, graphic bounds, transition, subtitle safe area, coverage note, and
human-review warnings.

Codex authors and reviews the plan. The registered `composition_quality` tool
checks timeline continuity, asset paths, crop geometry, protected-region and
graphic collisions, repeated sources/motifs, scene duration, transition and
animation bounds, contain-fit occupancy, and incomplete coverage. It cannot
determine semantic relevance or visual quality and always returns
`human_review_required`; errors block approval, warnings require inspection.

The existing Remotion `Explainer` accepts crop `mode`, normalized `focal_point`
and source dimensions, restrained image `motion_intensity`, scene-local
`graphics`, and transition names on existing cuts. `contain_blur` preserves
the full source frame over a blurred backplate. Scene graphics are separate
from `CaptionOverlay`; subtitle timing is not changed. A flat still has no
depth data, so motion is a pan/zoom rather than true parallax.

For production, the pipeline still requires an explicit `render_runtime`
decision at proposal time and a human-reviewed composition plan before render.
When Remotion and HyperFrames are both available, present both runtime choices
as required by `AGENT_GUIDE.md`; Stage 3C does not silently lock a runtime.
