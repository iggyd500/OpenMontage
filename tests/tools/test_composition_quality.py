from pathlib import Path

import pytest

from tools.video.composition_quality import (
    CompositionQuality,
    analyze_composition_plan,
    calculate_cover_window,
    calculate_contain_zoom_window,
)


def test_cover_window_for_landscape_vertical_frame_and_off_center_focal_point():
    window = calculate_cover_window(1920, 1080, 1080, 1920, focal_x=0.78, focal_y=0.5)
    assert window["width"] == pytest.approx(0.316406)
    assert window["height"] == 1
    assert window["x"] > 0.6
    assert 0 <= window["object_position_x"] <= 1


@pytest.mark.parametrize("dimensions", [(0, 10, 9, 16), (10, 0, 9, 16), (10, 10, 0, 16)])
def test_cover_window_rejects_nonpositive_dimensions(dimensions):
    with pytest.raises(ValueError):
        calculate_cover_window(*dimensions)


def test_cover_window_rejects_non_normalized_focal_point():
    with pytest.raises(ValueError):
        calculate_cover_window(1920, 1080, 1080, 1920, focal_x=1.2)


def test_contain_zoom_window_keeps_full_source_at_scale_one_and_centered_zoom_at_scale_1_2():
    full = calculate_contain_zoom_window(3840, 2160, 1080, 1920, scale=1.0)
    zoomed = calculate_contain_zoom_window(3840, 2160, 1080, 1920, scale=1.2)
    assert full == {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    assert zoomed["x"] == pytest.approx(1 / 12, abs=1e-6)
    assert zoomed["width"] == pytest.approx(5 / 6, abs=1e-6)
    assert zoomed["height"] == 1.0


def test_contain_zoom_quality_check_flags_clipped_protected_content():
    plan = {"profile": {"width": 1080, "height": 1920}, "scenes": [{
        "scene_id": "zoom", "timeline": {"start_seconds": 0, "end_seconds": 10},
        "source": {"asset_id": "a", "path": "missing.mp4", "width": 3840, "height": 2160, "kind": "video", "duration_seconds": 10},
        "crop": {"mode": "contain_blur", "scale": 1.25, "focal_point": {"x": .5, "y": .5},
                 "protected_regions": [{"id": "team-and-chart", "x": .05, "y": .1, "width": .9, "height": .8}]},
        "graphics": [], "transition": {"in_type": "cut"}, "coverage_status": "complete",
    }]}
    report = analyze_composition_plan(plan)
    assert any(item["code"] == "protected_region_clipped" for item in report["warnings"])


@pytest.mark.parametrize("scale", [0.99, 1.51, "not-a-number"])
def test_contain_zoom_rejects_invalid_scale(scale):
    with pytest.raises((TypeError, ValueError)):
        calculate_contain_zoom_window(3840, 2160, 1080, 1920, scale=scale)


def test_quality_report_flags_clip_timing_missing_source_and_graphic_collisions(tmp_path: Path):
    plan = {
        "profile": {"width": 1080, "height": 1920},
        "scenes": [
            {
                "scene_id": "one", "timeline": {"start_seconds": 0, "end_seconds": 10},
                "source": {"asset_id": "a", "path": "missing.mp4", "width": 1920, "height": 1080, "kind": "video", "duration_seconds": 5},
                    "crop": {"mode": "cover", "focal_point": {"x": .5, "y": .5}, "protected_regions": [{"id": "face", "x": .45, "y": .2, "width": .15, "height": .3}, {"id": "off-frame", "x": .75, "y": .2, "width": .15, "height": .3}]},
                "graphics": [{"id": "title", "region": {"x": .7, "y": .2, "width": .2, "height": .3}}],
                "transition": {"in_type": "fade"}, "coverage_status": "partial",
            },
            {
                "scene_id": "two", "timeline": {"start_seconds": 11, "end_seconds": 12},
                "source": {"asset_id": "a", "width": 1080, "height": 1920, "kind": "image"},
                "crop": {"mode": "cover", "focal_point": {"x": .5, "y": .5}, "protected_regions": []},
                "graphics": [], "transition": {"in_type": "fade"},
                "coverage_status": "insufficient",
            },
            {
                "scene_id": "three", "timeline": {"start_seconds": 12, "end_seconds": 13},
                "source": {"asset_id": "c", "width": 1080, "height": 1920, "kind": "image"},
                "crop": {"mode": "cover", "focal_point": {"x": .5, "y": .5}, "protected_regions": []},
                "graphics": [], "transition": {"in_type": "fade"},
                "coverage_status": "complete",
            },
        ],
    }
    report = analyze_composition_plan(plan, project_root=tmp_path)
    codes = {warning["code"] for warning in report["warnings"]}
    assert report["quality_status"] == "blocked"
    assert {"missing_source_file", "source_shorter_than_scene", "protected_region_clipped", "graphic_overlaps_subject", "timeline_gap", "duplicate_asset", "uniform_transitions", "incomplete_visual_support"}.issubset(codes)
    assert report["automatic_approval"] is False


def test_tool_is_discoverable_and_never_auto_approves():
    from tools.tool_registry import ToolRegistry

    registry = ToolRegistry()
    assert "composition_quality" in registry.discover()
    result = registry.get("composition_quality").execute({"composition_plan": {"scenes": []}})
    assert result.success
    assert result.data["automatic_approval"] is False


def test_quality_warnings_cover_caption_collision_animation_and_pacing():
    plan = {"profile": {"width": 1080, "height": 1920}, "scenes": [
        {"scene_id": "short", "timeline": {"start_seconds": 0, "end_seconds": 2},
         "source": {"asset_id": "a", "width": 1080, "height": 1920, "kind": "image"},
         "crop": {"mode": "cover", "focal_point": {"x": .5, "y": .5}, "protected_regions": []},
         "animation": {"intensity": .2}, "graphics": [{"id": "caption-overlap", "region": {"x": .1, "y": .8, "width": .6, "height": .1}}],
         "subtitle_safe_area": {"x": .08, "y": .78, "width": .84, "height": .16},
         "transition": {"in_type": "fade", "duration_frames": 40}},
        {"scene_id": "long", "timeline": {"start_seconds": 2, "end_seconds": 14},
         "source": {"asset_id": "b", "width": 1080, "height": 1920, "kind": "image"},
         "crop": {"mode": "cover", "focal_point": {"x": .5, "y": .5}, "protected_regions": []},
         "animation": {"intensity": .04}, "graphics": [], "transition": {"in_type": "cut", "duration_frames": 0}},
    ]}
    report = analyze_composition_plan(plan)
    codes = {item["code"] for item in report["warnings"]}
    assert {"graphic_overlaps_subtitle_safe_area", "invalid_animation_intensity", "transition_too_long", "scene_pacing_outlier"}.issubset(codes)
    assert report["quality_status"] == "blocked"
