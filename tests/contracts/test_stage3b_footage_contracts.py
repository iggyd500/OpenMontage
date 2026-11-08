import json
from pathlib import Path

from schemas.artifacts import list_schemas, validate_artifact
from lib.pipeline_loader import load_pipeline

ROOT = Path(__file__).resolve().parents[2]


def test_scene_visual_brief_schema_is_registered_and_requires_concrete_queries():
    assert "scene_visual_briefs" in list_schemas()
    artifact = {
        "version": "1.0",
        "scenes": [{
            "scene_id": "scene-1", "narration_text": "A doctor examines scans.",
            "main_subject": "Radiologist", "visual_message": "Clinical scan review supports medical decisions.",
            "relevant_objects": ["medical scan", "monitor"], "relevant_people": ["doctor"],
            "setting": "Hospital radiology room", "desired_action": "Doctor examining medical scans",
            "visual_mood": "calm and focused", "suggested_footage_type": "documentary_broll",
            "search_keywords": ["doctor examining medical scans", "radiologist reviewing scan display"],
            "exclusion_keywords": ["sci-fi hologram", "brand logos"],
            "media_approach": "stock_video", "fallback_media": ["stock_image", "motion_graphics"],
            "scene_duration_seconds": 12,
        }],
    }
    validate_artifact("scene_visual_briefs", artifact)
    artifact["scenes"][0]["search_keywords"] = ["AI technology", "future"]
    try:
        validate_artifact("scene_visual_briefs", artifact)
    except Exception:
        pass
    else:
        raise AssertionError("generic, undersized queries should fail the minimum-length constraint")


def test_footage_selection_plan_schema_records_unfilled_scene_without_fake_selection():
    assert "footage_selection_plan" in list_schemas()
    validate_artifact("footage_selection_plan", {
        "version": "1.0",
        "scenes": [{
            "scene_id": "scene-1", "visual_brief": {"scene_id": "scene-1"},
            "search_queries": [{
                "provider": "openmontage", "source": "pixabay_image",
                "query": "doctor with medical records", "candidate_count": 0,
                "note": "No metadata-only adapter is registered for this source.",
            }], "candidate_count": 0, "candidates": [],
            "selection_status": "no_suitable_footage", "selected_asset_id": None,
            "selected_rationale": "No relevant footage was found.", "rejected_candidates": [],
            "no_suitable_footage": True, "missing_reason": "All sources returned no matching assets.",
            "editorial_caveats": ["A designed graphic is needed."],
        }],
    })


def test_animated_explainer_requires_scene_briefs_and_exposes_stock_tools():
    manifest = load_pipeline("animated-explainer")
    stages = {stage["name"]: stage for stage in manifest["stages"]}
    assert "scene_visual_briefs" in stages["scene_plan"]["produces"]
    assert "scene_visual_briefs" in stages["assets"]["required_artifacts_in"]
    assert "footage_selection_plan" in stages["assets"]["produces"]
    assert {"stock_candidate_search", "stock_candidate_ranker", "stock_candidate_preview", "stock_candidate_download"}.issubset(
        set(stages["assets"]["required_tools"])
    )


def test_existing_everyday_ai_scene_plan_has_scene_ids_for_brief_mapping():
    scene_plan = json.loads((ROOT / "projects/everyday-ai-first-production/artifacts/scene_plan.json").read_text(encoding="utf-8"))
    script = json.loads((ROOT / "projects/everyday-ai-first-production/artifacts/script.json").read_text(encoding="utf-8"))
    prompt_data = json.loads((ROOT / "projects/everyday-ai-first-production/artifacts/visual_prompts.json").read_text(encoding="utf-8"))
    scene_ids = {scene["id"] for scene in scene_plan["scenes"]}
    section_ids = {section["id"] for section in script["sections"]}
    assert len(scene_ids) == len(scene_plan["scenes"]) == 5
    assert all(scene.get("script_section_id") in section_ids for scene in scene_plan["scenes"])
    assert {prompt["scene_id"] for prompt in prompt_data["prompts"]} == scene_ids


def test_new_stock_capabilities_are_registered():
    from tools.tool_registry import ToolRegistry

    registry = ToolRegistry()
    discovered = registry.discover()
    assert {"stock_candidate_search", "stock_candidate_ranker", "stock_candidate_preview", "stock_candidate_download"}.issubset(discovered)
