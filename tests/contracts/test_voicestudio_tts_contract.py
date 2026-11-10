from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _manifest(name: str) -> dict:
    with (ROOT / "pipeline_defs" / f"{name}.yaml").open(encoding="utf-8") as source:
        return yaml.safe_load(source)


def test_explainer_and_localization_assets_use_tts_selector():
    for pipeline in ("animated-explainer", "localization-dub"):
        manifest = _manifest(pipeline)
        asset_stage = next(stage for stage in manifest["stages"] if stage["name"] == "assets")
        assert "tts_selector" in asset_stage.get("required_tools", [])
        assert "tts_selector" in asset_stage.get("tools_available", [])


def test_director_instructions_document_explicit_voicestudio_routing():
    explainer = (ROOT / "skills/pipelines/explainer/asset-director.md").read_text(encoding="utf-8")
    localization = (ROOT / "skills/pipelines/localization-dub/asset-director.md").read_text(encoding="utf-8")
    layer3 = (ROOT / ".agents/skills/voicestudio-tts/SKILL.md").read_text(encoding="utf-8")

    for content in (explainer, localization, layer3):
        assert "voicestudio_tts" in content
        assert "expr-voice-2-m" in content
        assert "fallback" in content or "switch" in content
