"""Transparent metadata-based stock candidate scoring.

This is a ranking aid, not an autonomous editor or a visual model. The
orchestrating agent must inspect selected previews/frames and explicitly
approve an asset before it is treated as selected.
"""
from __future__ import annotations

import re
import time
from typing import Any

from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile, ToolResult,
    ToolRuntime, ToolStability, ToolStatus, ToolTier,
)

_STOP = {"a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into", "is", "of", "on", "or", "the", "to", "with", "using", "person", "people"}
_WEIGHTS = {
    "semantic_relevance": 0.30,
    "subject_object_match": 0.15,
    "action_match": 0.15,
    "environment_match": 0.10,
    "resolution_suitability": 0.15,
    "aspect_ratio_suitability": 0.10,
    "duration_suitability": 0.05,
}


class StockCandidateRanker(BaseTool):
    name = "stock_candidate_ranker"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "stock_candidate_evaluation"
    provider = "openmontage"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    capabilities = ["metadata_relevance_scoring", "candidate_diversification", "explicit_asset_selection"]
    supports = {"transparent_component_scores": True, "uses_visual_model": False, "requires_agent_visual_review": True}
    best_for = ["ranking stock candidates against a scene visual brief", "surfacing duplicates and technical fit"]
    dependencies = []
    side_effects = []
    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=64, vram_mb=0, disk_mb=0, network_required=False)

    input_schema = {
        "type": "object",
        "required": ["scene_visual_brief", "candidates"],
        "properties": {
            "scene_visual_brief": {"type": "object"},
            "candidates": {"type": "array", "items": {"type": "object"}},
            "scene_duration_seconds": {"type": "number", "minimum": 0},
            "target_aspect_ratio": {"type": "string", "default": "9:16"},
            "target_width": {"type": "integer", "default": 1080},
            "target_height": {"type": "integer", "default": 1920},
            "previously_selected": {"type": "array", "items": {"type": "object"}},
            "approved_asset_id": {"type": "string", "description": "Only pass after Codex/human has reviewed the actual candidate imagery."},
        },
    }

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        brief = inputs["scene_visual_brief"]
        candidates = inputs["candidates"]
        if not isinstance(brief, dict) or not isinstance(candidates, list):
            return ToolResult(success=False, error="scene_visual_brief must be an object and candidates must be an array")
        ranked = rank_candidates(
            brief,
            candidates,
            scene_duration_seconds=inputs.get("scene_duration_seconds"),
            target_aspect_ratio=inputs.get("target_aspect_ratio", "9:16"),
            target_width=int(inputs.get("target_width", 1080)),
            target_height=int(inputs.get("target_height", 1920)),
            previously_selected=inputs.get("previously_selected") or [],
        )
        approved_id = inputs.get("approved_asset_id")
        selected = None
        selection_status = "review_required" if ranked else "no_candidates"
        if approved_id:
            selected, selection_status = approve_candidate(approved_id, ranked)
        return ToolResult(
            success=True,
            data={
                "scene_id": brief.get("scene_id"),
                "candidate_count": len(ranked),
                "ranked_candidates": ranked,
                "recommended_candidate_id": ranked[0]["asset_id"] if ranked else None,
                "selection_status": selection_status,
                "selected_candidate": selected,
                "score_basis": "metadata_keyword_match_and_technical_fit; not visual analysis",
                "weights": _WEIGHTS,
                "visual_confirmation_required": True,
                "selection_policy": "Ranking alone does not select an asset. Pass approved_asset_id only after inspecting the candidate's thumbnail/frame; reused asset IDs and source URLs are rejected.",
            },
            cost_usd=0.0,
            duration_seconds=round(time.perf_counter() - started, 3),
        )


def rank_candidates(
    brief: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    scene_duration_seconds: float | None = None,
    target_aspect_ratio: str = "9:16",
    target_width: int = 1080,
    target_height: int = 1920,
    previously_selected: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Score each result with explicit dimensions and no hidden cut-off."""
    keywords = brief.get("search_keywords") or []
    prior = previously_selected or []
    scored = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        text = " ".join(str(candidate.get(key) or "") for key in ("description", "source_tags", "tags"))
        text += " " + " ".join(str(x) for x in (candidate.get("tags") or []) if isinstance(x, str))
        semantic_terms = list(keywords) + [str(brief.get("visual_message") or "")]
        semantic = _weighted_term_coverage(semantic_terms, text)
        subject = _coverage(_terms(brief.get("main_subject")) | _terms(brief.get("relevant_objects")), text)
        action = _coverage(_terms(brief.get("desired_action")), text)
        setting = _coverage(_terms(brief.get("setting")), text)
        width = int((candidate.get("resolution") or {}).get("width") or 0)
        height = int((candidate.get("resolution") or {}).get("height") or 0)
        pixels = (width * height) ** 0.5 if width and height else 0
        target_pixels = max(1, (target_width * target_height) ** 0.5)
        resolution = min(1.0, pixels / target_pixels) if pixels else 0.0
        ratio = _parse_ratio(target_aspect_ratio)
        candidate_ratio = width / height if width and height else None
        aspect = _ratio_score(candidate_ratio, ratio)
        duration = candidate.get("duration_seconds")
        kind = candidate.get("kind", "video")
        if kind == "image":
            duration_score, duration_state = 1.0, "not_applicable"
        elif duration is None or duration <= 0:
            duration_score, duration_state = 0.0, "unknown"
        elif scene_duration_seconds is None or scene_duration_seconds <= 0:
            duration_score, duration_state = 1.0, "available_not_compared"
        else:
            duration_score = min(1.0, float(duration) / float(scene_duration_seconds))
            duration_state = "sufficient" if duration >= scene_duration_seconds else "shorter_than_scene"

        components = {
            "semantic_relevance": round(semantic, 4),
            "subject_object_match": round(subject, 4),
            "action_match": round(action, 4),
            "environment_match": round(setting, 4),
            "resolution_suitability": round(resolution, 4),
            "aspect_ratio_suitability": round(aspect, 4),
            "duration_suitability": round(duration_score, 4),
        }
        raw_score = sum(components[key] * weight for key, weight in _WEIGHTS.items())
        duplicate, repetition = _duplicate_and_repetition(candidate, prior)
        penalty = 0.15 * repetition
        final_score = max(0.0, raw_score - penalty)
        scored.append({
            **candidate,
            "score": round(final_score, 4),
            "score_percent": round(final_score * 100, 1),
            "score_components": components,
            "repetition_penalty": round(penalty, 4),
            "duplicate_of_previous_selection": duplicate,
            "duration_status": duration_state,
            "assessment_basis": "metadata_only; provider title/tags and technical metadata do not prove depicted content",
            "visual_review_required": True,
            "visual_review_status": "not_reviewed",
        })
    scored.sort(key=lambda item: (item["duplicate_of_previous_selection"], -item["score"], str(item.get("asset_id", ""))))
    return scored


def approve_candidate(asset_id: str, ranked: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """Approve a candidate only when the orchestrator explicitly names it."""
    for candidate in ranked:
        if candidate.get("asset_id") != asset_id:
            continue
        if candidate.get("duplicate_of_previous_selection"):
            return None, "duplicate_rejected"
        return candidate, "approved_by_orchestrator_after_visual_review"
    return None, "approved_candidate_not_found"


def _terms(value: Any) -> set[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = [str(item) for item in value]
    else:
        values = []
    return {token for item in values for token in _tokens(item)}


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.casefold()) if len(token) > 2 and token not in _STOP}


def _coverage(terms: set[str], text: str) -> float:
    if not terms:
        return 0.0
    available = _tokens(text)
    return len(terms & available) / len(terms)


def _weighted_term_coverage(groups: list[Any], text: str) -> float:
    valid = [str(group) for group in groups if str(group).strip()]
    if not valid:
        return 0.0
    # Average phrase-level token coverage so a long summary cannot swamp the
    # concise concrete search phrases.
    scores = [_coverage(_tokens(group), text) for group in valid]
    return sum(scores) / len(scores)


def _parse_ratio(value: str) -> float:
    try:
        numerator, denominator = value.split(":", 1)
        result = float(numerator) / float(denominator)
        return result if result > 0 else 9 / 16
    except (ValueError, ZeroDivisionError, AttributeError):
        return 9 / 16


def _ratio_score(actual: float | None, target: float) -> float:
    if actual is None or actual <= 0:
        return 0.0
    return max(0.0, 1.0 - min(abs(__import__("math").log2(actual / target)), 1.0))


def _duplicate_and_repetition(candidate: dict[str, Any], previous: list[dict[str, Any]]) -> tuple[bool, float]:
    candidate_id = str(candidate.get("asset_id") or "")
    candidate_url = str(candidate.get("source_url") or "")
    candidate_terms = _tokens(" ".join(str(candidate.get(key) or "") for key in ("description", "source_tags", "tags")))
    duplicate = False
    max_similarity = 0.0
    for selected in previous:
        if candidate_id and candidate_id == str(selected.get("asset_id") or ""):
            duplicate = True
        if candidate_url and candidate_url == str(selected.get("source_url") or ""):
            duplicate = True
        selected_terms = _tokens(" ".join(str(selected.get(key) or "") for key in ("description", "source_tags", "tags")))
        union = candidate_terms | selected_terms
        if union:
            max_similarity = max(max_similarity, len(candidate_terms & selected_terms) / len(union))
    return duplicate, max_similarity
