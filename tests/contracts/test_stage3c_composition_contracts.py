import json
from pathlib import Path

from lib.pipeline_loader import load_pipeline
from schemas.artifacts import list_schemas, validate_artifact

ROOT = Path(__file__).resolve().parents[2]


def test_composition_plan_schema_and_pipeline_gate_are_registered():
    assert "composition_plan" in list_schemas()
    stages = {stage["name"]: stage for stage in load_pipeline("animated-explainer")["stages"]}
    assert "composition_plan" in stages["edit"]["produces"]
    assert "composition_quality" in stages["edit"]["required_tools"]
    assert "composition_plan" in stages["compose"]["required_artifacts_in"]
    assert stages["edit"]["human_approval_default"] is True


def test_everyday_ai_stage3c_plan_has_five_narration_linked_scene_specs():
    plan_path = ROOT / "projects/everyday-ai-first-production/artifacts/stage3c-validation/composition-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_artifact("composition_plan", plan)
    assert len(plan["scenes"]) == 5
    assert all(scene["narration_text"] and scene["crop"]["focal_point"] for scene in plan["scenes"])
    assert all(scene["animation"]["parallax_applied"] is False for scene in plan["scenes"])
    assert {scene["transition"]["in_type"] for scene in plan["scenes"]} > {"cut"}
    assert plan["human_review_required"] is True


def test_renderer_explainer_has_scene_composition_features_without_caption_changes():
    source = (ROOT / "remotion-composer/src/Explainer.tsx").read_text(encoding="utf-8")
    assert "getObjectPosition" in source
    assert 'mode === "contain_blur"' in source
    assert "SceneGraphics" in source
    assert "SceneTransition" in source
    assert "CaptionOverlay" in source
