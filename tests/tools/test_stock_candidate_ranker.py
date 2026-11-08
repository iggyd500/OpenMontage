from tools.video.stock_candidate_ranker import (
    StockCandidateRanker, approve_candidate, rank_candidates,
)


def _brief():
    return {
        "scene_id": "scene-health",
        "main_subject": "doctor",
        "visual_message": "doctor examines medical scans on a diagnostic display",
        "relevant_objects": ["medical scans", "computer monitor"],
        "desired_action": "examining scans",
        "setting": "hospital radiology room",
        "search_keywords": ["doctor examining medical scans", "radiologist reviewing scan on monitor"],
    }


def _candidate(asset_id, tags, *, width=720, height=1280, duration=14, url=None):
    return {
        "asset_id": asset_id, "source_url": url or f"https://provider/item/{asset_id}",
        "source_tags": tags, "description": tags, "tags": tags.split(", "),
        "kind": "video", "duration_seconds": duration,
        "resolution": {"width": width, "height": height},
    }


def test_relevance_score_ranks_scene_specific_content_and_explains_components():
    ranked = rank_candidates(_brief(), [
        _candidate("weak", "technology, abstract digital network"),
        _candidate("match", "doctor, radiologist examining medical scans on monitor, hospital radiology room"),
    ], scene_duration_seconds=12)
    assert ranked[0]["asset_id"] == "match"
    assert set(ranked[0]["score_components"]) == {
        "semantic_relevance", "subject_object_match", "action_match",
        "environment_match", "resolution_suitability",
        "aspect_ratio_suitability", "duration_suitability",
    }
    assert ranked[0]["score_percent"] > ranked[1]["score_percent"]
    assert ranked[0]["assessment_basis"].startswith("metadata_only")
    assert ranked[0]["visual_review_required"] is True


def test_aspect_ratio_and_resolution_suitability():
    vertical = rank_candidates(_brief(), [_candidate("v", "doctor medical scans")], target_aspect_ratio="9:16")[0]
    landscape = rank_candidates(_brief(), [_candidate("h", "doctor medical scans", width=1920, height=1080)], target_aspect_ratio="9:16")[0]
    assert vertical["score_components"]["aspect_ratio_suitability"] == 1.0
    assert landscape["score_components"]["aspect_ratio_suitability"] < vertical["score_components"]["aspect_ratio_suitability"]
    assert vertical["score_components"]["resolution_suitability"] < 1.0


def test_duration_suitability_flags_short_and_unknown_clips():
    ranked = rank_candidates(_brief(), [
        _candidate("long", "doctor scan", duration=15),
        _candidate("short", "doctor scan", duration=4),
        _candidate("unknown", "doctor scan", duration=None),
    ], scene_duration_seconds=12)
    by_id = {row["asset_id"]: row for row in ranked}
    assert by_id["long"]["duration_status"] == "sufficient"
    assert by_id["short"]["duration_status"] == "shorter_than_scene"
    assert by_id["unknown"]["duration_status"] == "unknown"
    assert by_id["long"]["score_components"]["duration_suitability"] == 1.0
    assert by_id["short"]["score_components"]["duration_suitability"] < 1.0


def test_duplicate_and_repetitive_candidates_are_flagged_and_penalized():
    rows = rank_candidates(
        _brief(), [_candidate("used", "doctor medical scans", url="https://provider/same")],
        previously_selected=[_candidate("used", "doctor medical scans", url="https://provider/same")],
    )
    assert rows[0]["duplicate_of_previous_selection"] is True
    selected, status = approve_candidate("used", rows)
    assert selected is None
    assert status == "duplicate_rejected"


def test_selection_requires_explicit_approval_and_missing_footage_is_not_forced():
    rows = rank_candidates(_brief(), [_candidate("candidate", "doctor scans")])
    assert approve_candidate("candidate", rows)[0] is not None
    assert approve_candidate("other", rows) == (None, "approved_candidate_not_found")
    result = StockCandidateRanker().execute({"scene_visual_brief": _brief(), "candidates": []})
    assert result.data["selection_status"] == "no_candidates"
    assert result.data["selected_candidate"] is None


def test_image_duration_is_not_penalized():
    candidate = _candidate("still", "doctor medical scan", duration=None)
    candidate["kind"] = "image"
    row = rank_candidates(_brief(), [candidate], scene_duration_seconds=12)[0]
    assert row["duration_status"] == "not_applicable"
    assert row["score_components"]["duration_suitability"] == 1.0
