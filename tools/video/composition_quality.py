"""Deterministic technical checks for agent-authored scene composition plans.

The tool reports geometry, timeline, file, and repetition risks. It does not
select assets, judge aesthetics, or approve a composition; Codex owns those
decisions and every scene remains subject to human review.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile, ToolResult,
    ToolRuntime, ToolStability, ToolStatus, ToolTier,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def calculate_cover_window(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    focal_x: float = 0.5,
    focal_y: float = 0.5,
) -> dict[str, float]:
    """Return normalized source crop and CSS object-position for object-fit cover."""
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Source and target dimensions must be positive")
    if not (0 <= focal_x <= 1 and 0 <= focal_y <= 1):
        raise ValueError("Focal coordinates must be normalized to [0, 1]")

    source_ratio = source_width / source_height
    target_ratio = target_width / target_height
    if source_ratio > target_ratio:
        visible_width = target_ratio / source_ratio
        visible_height = 1.0
    else:
        visible_width = 1.0
        visible_height = source_ratio / target_ratio

    left = _clamp(focal_x - visible_width / 2, 0.0, 1.0 - visible_width)
    top = _clamp(focal_y - visible_height / 2, 0.0, 1.0 - visible_height)
    object_x = left / (1.0 - visible_width) if visible_width < 1 else 0.5
    object_y = top / (1.0 - visible_height) if visible_height < 1 else 0.5
    return {
        "x": round(left, 6), "y": round(top, 6),
        "width": round(visible_width, 6), "height": round(visible_height, 6),
        "object_position_x": round(object_x, 6),
        "object_position_y": round(object_y, 6),
    }


def calculate_contain_zoom_window(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    scale: float = 1.0,
    focal_x: float = 0.5,
    focal_y: float = 0.5,
) -> dict[str, float]:
    """Return the source area visible after a restrained zoom of contained media."""
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Source and target dimensions must be positive")
    if not (1.0 <= scale <= 1.5):
        raise ValueError("Contained media scale must be between 1.0 and 1.5")
    if not (0 <= focal_x <= 1 and 0 <= focal_y <= 1):
        raise ValueError("Focal coordinates must be normalized to [0, 1]")

    source_ratio = source_width / source_height
    target_ratio = target_width / target_height
    base_width_occupancy = min(1.0, source_ratio / target_ratio)
    base_height_occupancy = min(1.0, target_ratio / source_ratio)
    visible_width = min(1.0, 1.0 / (scale * base_width_occupancy))
    visible_height = min(1.0, 1.0 / (scale * base_height_occupancy))
    left = _clamp(focal_x - visible_width / 2, 0.0, 1.0 - visible_width)
    top = _clamp(focal_y - visible_height / 2, 0.0, 1.0 - visible_height)
    return {
        "x": round(left, 6), "y": round(top, 6),
        "width": round(visible_width, 6), "height": round(visible_height, 6),
    }


def _inside(rect: dict[str, Any], window: dict[str, float], tolerance: float = 1e-6) -> bool:
    x, y = float(rect.get("x", 0)), float(rect.get("y", 0))
    w, h = float(rect.get("width", 0)), float(rect.get("height", 0))
    return (
        x >= window["x"] - tolerance
        and y >= window["y"] - tolerance
        and x + w <= window["x"] + window["width"] + tolerance
        and y + h <= window["y"] + window["height"] + tolerance
    )


def _overlap_fraction(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay, aw, ah = (float(a.get(k, 0)) for k in ("x", "y", "width", "height"))
    bx, by, bw, bh = (float(b.get(k, 0)) for k in ("x", "y", "width", "height"))
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    area = max(0.0, aw) * max(0.0, ah)
    return (iw * ih / area) if area else 0.0


def _protected_region_on_canvas(
    region: dict[str, Any], *, mode: str, window: dict[str, float] | None,
    source_width: int, source_height: int, target_width: int, target_height: int,
    scale: float = 1.0,
) -> dict[str, float] | None:
    """Map a source-normalized subject box into output-canvas coordinates."""
    if mode == "cover":
        if not window:
            return None
        x0, y0 = window["x"], window["y"]
        width, height = window["width"], window["height"]
        return {
            "x": (float(region["x"]) - x0) / width,
            "y": (float(region["y"]) - y0) / height,
            "width": float(region["width"]) / width,
            "height": float(region["height"]) / height,
        }
    source_ratio, target_ratio = source_width / source_height, target_width / target_height
    if scale > 1 and window:
        frame_width = min(1.0, min(1.0, source_ratio / target_ratio) * scale)
        frame_height = min(1.0, min(1.0, target_ratio / source_ratio) * scale)
        return {
            "x": (1 - frame_width) / 2 + (float(region["x"]) - window["x"]) / window["width"] * frame_width,
            "y": (1 - frame_height) / 2 + (float(region["y"]) - window["y"]) / window["height"] * frame_height,
            "width": float(region["width"]) / window["width"] * frame_width,
            "height": float(region["height"]) / window["height"] * frame_height,
        }
    if source_ratio > target_ratio:
        frame = {"x": 0.0, "y": (1 - target_ratio / source_ratio) / 2, "width": 1.0, "height": target_ratio / source_ratio}
    else:
        frame = {"x": (1 - source_ratio / target_ratio) / 2, "y": 0.0, "width": source_ratio / target_ratio, "height": 1.0}
    return {
        "x": frame["x"] + float(region["x"]) * frame["width"],
        "y": frame["y"] + float(region["y"]) * frame["height"],
        "width": float(region["width"]) * frame["width"],
        "height": float(region["height"]) * frame["height"],
    }


class CompositionQuality(BaseTool):
    name = "composition_quality"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "composition_quality_control"
    provider = "openmontage"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    capabilities = ["scene_crop_geometry", "timeline_validation", "composition_warnings"]
    supports = {"metadata_only": True, "never_auto_approves": True, "requires_human_review": True}
    best_for = ["checking scene crops and timing before render", "flagging overlay collisions and repeated sources"]
    dependencies = []
    side_effects = []
    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=64, vram_mb=0, disk_mb=0, network_required=False)

    input_schema = {
        "type": "object",
        "required": ["composition_plan"],
        "properties": {
            "composition_plan": {"type": "object"},
            "project_root": {"type": "string"},
            "crop_warning_fraction": {"type": "number", "default": 0.4},
            "overlap_warning_fraction": {"type": "number", "default": 0.15},
        },
    }

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        plan = inputs.get("composition_plan")
        if not isinstance(plan, dict) or not isinstance(plan.get("scenes"), list):
            return ToolResult(success=False, error="composition_plan must contain a scenes array")
        report = analyze_composition_plan(
            plan,
            project_root=Path(inputs["project_root"]).resolve() if inputs.get("project_root") else None,
            crop_warning_fraction=float(inputs.get("crop_warning_fraction", 0.4)),
            overlap_warning_fraction=float(inputs.get("overlap_warning_fraction", 0.15)),
        )
        return ToolResult(
            success=True,
            data=report,
            cost_usd=0.0,
            duration_seconds=round(time.perf_counter() - started, 4),
        )


def analyze_composition_plan(
    plan: dict[str, Any],
    *,
    project_root: Path | None = None,
    crop_warning_fraction: float = 0.4,
    overlap_warning_fraction: float = 0.15,
) -> dict[str, Any]:
    scenes = plan.get("scenes") or []
    warnings: list[dict[str, Any]] = []
    per_scene: list[dict[str, Any]] = []
    previous_end: float | None = None
    seen_assets: set[str] = set()
    prior_motifs: set[str] = set()
    transition_types: list[str] = []

    def warn(scene_id: str, code: str, message: str, severity: str = "warning") -> None:
        item = {"scene_id": scene_id, "code": code, "severity": severity, "message": message}
        warnings.append(item)
        per_scene_by_id[scene_id]["warnings"].append(item)

    per_scene_by_id: dict[str, dict[str, Any]] = {}
    for scene in sorted(scenes, key=lambda s: float((s.get("timeline") or {}).get("start_seconds", 0))):
        sid = str(scene.get("scene_id") or "unknown")
        row = {"scene_id": sid, "crop_window": None, "warnings": []}
        per_scene_by_id[sid] = row
        per_scene.append(row)
        timeline = scene.get("timeline") or {}
        start = float(timeline.get("start_seconds", 0))
        end = float(timeline.get("end_seconds", start))
        if end <= start:
            warn(sid, "invalid_scene_timing", "Scene end must be later than its start.", "error")
        if previous_end is not None:
            delta = start - previous_end
            if delta > 0.05:
                warn(sid, "timeline_gap", f"Timeline has a {delta:.3f}s gap before this scene.", "error")
            elif delta < -0.05:
                warn(sid, "timeline_overlap", f"Scene overlaps the previous scene by {-delta:.3f}s; verify intentional transition overlap.")
        previous_end = max(previous_end or end, end)

        source = scene.get("source") or {}
        asset_id = str(source.get("asset_id") or "")
        if asset_id and asset_id in seen_assets:
            warn(sid, "duplicate_asset", f"Source asset {asset_id} is reused; review for visual repetition.")
        if asset_id:
            seen_assets.add(asset_id)
        path = source.get("path")
        if project_root is not None and path:
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = project_root / resolved
            if not resolved.is_file():
                warn(sid, "missing_source_file", f"Composition source does not exist: {resolved}", "error")

        motif = str(scene.get("visual_motif") or "").strip().casefold()
        if motif and motif in prior_motifs:
            warn(sid, "repeated_visual_motif", f"Visual motif '{motif}' has already appeared; vary subject, framing, or graphic treatment.")
        prior_motifs.add(motif)

        width, height = int(source.get("width") or 0), int(source.get("height") or 0)
        profile = plan.get("profile") or {}
        out_w, out_h = int(profile.get("width") or 1080), int(profile.get("height") or 1920)
        crop = scene.get("crop") or {}
        mode = crop.get("mode", "cover")
        crop_window = None
        canvas_protected_regions: list[dict[str, Any]] = []
        if width <= 0 or height <= 0:
            warn(sid, "missing_source_dimensions", "Source dimensions are required to validate the crop.", "error")
        elif mode == "cover":
            focal = crop.get("focal_point") or {"x": 0.5, "y": 0.5}
            try:
                crop_window = calculate_cover_window(width, height, out_w, out_h, float(focal["x"]), float(focal["y"]))
            except (KeyError, TypeError, ValueError) as exc:
                warn(sid, "invalid_crop", f"Cover crop settings are invalid: {exc}", "error")
            if crop_window:
                row["crop_window"] = crop_window
                if crop_window["width"] < crop_warning_fraction:
                    warn(sid, "tight_horizontal_crop", f"Cover crop retains {crop_window['width']:.1%} of source width; review faces, hands, and objects in the actual preview.")
                if crop_window["height"] < crop_warning_fraction:
                    warn(sid, "tight_vertical_crop", f"Cover crop retains {crop_window['height']:.1%} of source height; review protected content in the actual preview.")
                for region in crop.get("protected_regions") or []:
                    if not _inside(region, crop_window):
                        warn(sid, "protected_region_clipped", f"Protected region '{region.get('id', 'unnamed')}' does not fit in the crop window.", "error")
                    mapped = _protected_region_on_canvas(region, mode=mode, window=crop_window, source_width=width, source_height=height, target_width=out_w, target_height=out_h)
                    if mapped:
                        canvas_protected_regions.append({**mapped, "id": region.get("id", "unnamed")})
        elif mode in {"contain_blur", "contain"}:
            try:
                media_scale = float(crop.get("scale", 1.0))
            except (TypeError, ValueError):
                media_scale = 0.0
            if not 1.0 <= media_scale <= 1.5:
                warn(sid, "invalid_contain_scale", "Contained media scale must be between 1.0 and 1.5.", "error")
                media_scale = 1.0
            if width and height:
                source_ratio, output_ratio = width / height, out_w / out_h
                base_occupancy = min(source_ratio / output_ratio, output_ratio / source_ratio)
                occupancy = min(1.0, base_occupancy * media_scale)
                row["content_frame_occupancy"] = round(occupancy, 4)
                if media_scale > 1:
                    try:
                        focal = crop.get("focal_point") or {"x": 0.5, "y": 0.5}
                        crop_window = calculate_contain_zoom_window(
                            width, height, out_w, out_h, media_scale,
                            float(focal["x"]), float(focal["y"]),
                        )
                        row["contained_zoom_window"] = crop_window
                        for region in crop.get("protected_regions") or []:
                            if not _inside(region, crop_window):
                                warn(sid, "protected_region_clipped", f"Zoomed contain framing clips protected content '{region.get('id', 'unnamed')}'.", "error")
                            mapped = _protected_region_on_canvas(
                                region, mode=mode, window=crop_window,
                                source_width=width, source_height=height,
                                target_width=out_w, target_height=out_h,
                                scale=media_scale,
                            )
                            if mapped:
                                canvas_protected_regions.append({**mapped, "id": region.get("id", "unnamed")})
                    except (KeyError, TypeError, ValueError) as exc:
                        warn(sid, "invalid_crop", f"Contained zoom settings are invalid: {exc}", "error")
                else:
                    for region in crop.get("protected_regions") or []:
                        mapped = _protected_region_on_canvas(region, mode=mode, window=None, source_width=width, source_height=height, target_width=out_w, target_height=out_h)
                        if mapped:
                            canvas_protected_regions.append({**mapped, "id": region.get("id", "unnamed")})
                if occupancy < 0.5:
                    warn(sid, "small_contained_frame", f"Contained media occupies about {occupancy:.1%} of the canvas; inspect legibility and background treatment.")
            if mode == "contain_blur":
                row["blurred_backplate"] = True
        else:
            warn(sid, "unsupported_fit_mode", f"Unsupported crop mode: {mode}", "error")

        source_duration = source.get("duration_seconds")
        if source.get("kind") == "video" and source_duration is not None and float(source_duration) < end - start:
            if not scene.get("allow_loop", False):
                warn(sid, "source_shorter_than_scene", f"Source clip ({float(source_duration):.2f}s) is shorter than this scene ({end-start:.2f}s); define a hold, loop, or alternate cut.")

        for graphic in scene.get("graphics") or []:
            rect = graphic.get("region") or {}
            x, y, w, h = (float(rect.get(k, -1)) for k in ("x", "y", "width", "height"))
            if min(x, y, w, h) < 0 or x + w > 1 or y + h > 1 or w == 0 or h == 0:
                warn(sid, "graphic_out_of_bounds", f"Graphic '{graphic.get('id', 'unnamed')}' is outside normalized canvas bounds.", "error")
                continue
            for region in canvas_protected_regions:
                if _overlap_fraction(rect, region) > overlap_warning_fraction:
                    warn(sid, "graphic_overlaps_subject", f"Graphic '{graphic.get('id', 'unnamed')}' may cover protected content '{region.get('id', 'unnamed')}'.")
            safe_area = scene.get("subtitle_safe_area")
            if safe_area and _overlap_fraction(rect, safe_area) > overlap_warning_fraction:
                warn(sid, "graphic_overlaps_subtitle_safe_area", f"Graphic '{graphic.get('id', 'unnamed')}' may collide with subtitles in the reserved safe area.")

        if not (scene.get("graphics") or []) and scene.get("coverage_status") != "complete":
            warn(sid, "no_graphic_visual_support", "Scene has incomplete visual coverage and no explanatory graphic support.")

        trans = scene.get("transition") or {}
        if trans.get("in_type"):
            transition_types.append(str(trans["in_type"]))
        if trans.get("duration_frames", 0) > 30:
            warn(sid, "transition_too_long", "Transition exceeds the 30-frame short-form limit; review its effect on narration timing.", "error")
        animation = scene.get("animation") or {}
        intensity = animation.get("intensity")
        if intensity is not None and not 0 <= float(intensity) <= 0.12:
            warn(sid, "invalid_animation_intensity", "Animation intensity must be between 0 and 0.12.", "error")
        quality = scene.get("coverage_status")
        if quality in {"partial", "insufficient"}:
            warn(sid, "incomplete_visual_support", str(scene.get("coverage_note") or "Scene visual support is incomplete."))

    if len(transition_types) >= 3 and len(set(transition_types)) == 1:
        warnings.append({"scene_id": None, "code": "uniform_transitions", "severity": "warning", "message": "Every scene uses the same transition; confirm this repetition is intentional."})

    durations = [float((s.get("timeline") or {}).get("end_seconds", 0)) - float((s.get("timeline") or {}).get("start_seconds", 0)) for s in scenes]
    positive_durations = sorted(d for d in durations if d > 0)
    if positive_durations:
        median = positive_durations[len(positive_durations) // 2]
        for scene, duration in zip(scenes, durations):
            if duration > 2 * median or duration < median / 2:
                sid = str(scene.get("scene_id") or "unknown")
                row = per_scene_by_id.get(sid)
                item = {"scene_id": sid, "code": "scene_pacing_outlier", "severity": "warning", "message": f"Scene duration ({duration:.2f}s) differs substantially from the median scene duration ({median:.2f}s); review pacing."}
                warnings.append(item)
                if row is not None:
                    row["warnings"].append(item)

    has_error = any(item["severity"] == "error" for item in warnings)
    return {
        "version": "1.0",
        "scene_count": len(scenes),
        "quality_status": "blocked" if has_error else "human_review_required",
        "technical_errors": sum(item["severity"] == "error" for item in warnings),
        "warning_count": sum(item["severity"] == "warning" for item in warnings),
        "scenes": per_scene,
        "warnings": warnings,
        "automatic_approval": False,
        "note": "Technical checks cannot establish visual quality. Review each preview, crop, text placement, and narration match before rendering.",
    }
