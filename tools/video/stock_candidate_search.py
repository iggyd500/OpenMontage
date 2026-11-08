"""Metadata-only discovery over OpenMontage's existing stock adapters.

This tool searches one named adapter at a time and never downloads search
results. The agent owns provider order, scene interpretation, shortlist review,
and the decision to call the separate selected-asset downloader.
"""
from __future__ import annotations

import time
from typing import Any

from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile, ToolResult,
    ToolRuntime, ToolStability, ToolStatus, ToolTier,
)


class StockCandidateSearch(BaseTool):
    name = "stock_candidate_search"
    version = "0.1.0"
    tier = ToolTier.SOURCE
    capability = "stock_discovery"
    provider = "openmontage"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.HYBRID

    dependencies = ["python:requests"]
    capabilities = ["metadata_only_stock_search", "stock_candidate_metadata"]
    supports = {"video_and_image": True, "downloads_search_results": False}
    best_for = ["shortlisting provider results before downloading", "scene-specific stock discovery"]
    side_effects = ["calls the selected stock provider search API"]
    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=128, vram_mb=0, disk_mb=0, network_required=True)

    input_schema = {
        "type": "object",
        "required": ["source", "query"],
        "properties": {
            "source": {"type": "string", "description": "Existing stock_sources adapter name, e.g. pixabay_video or coverr."},
            "query": {"type": "string", "minLength": 2},
            "kind": {"type": "string", "enum": ["video", "image"], "default": "video"},
            "per_page": {"type": "integer", "default": 8, "minimum": 1, "maximum": 20},
            "page": {"type": "integer", "default": 1, "minimum": 1},
            "min_duration": {"type": "number", "minimum": 0},
            "max_duration": {"type": "number", "minimum": 0},
            "orientation": {"type": "string", "enum": ["portrait", "landscape", "square"]},
            "min_width": {"type": "integer", "minimum": 1},
        },
    }

    def get_status(self) -> ToolStatus:
        try:
            from tools.video.stock_sources import available_sources
            return ToolStatus.AVAILABLE if available_sources() else ToolStatus.UNAVAILABLE
        except Exception:
            return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        from tools.video.stock_sources import SearchFilters, get_source

        source_name = inputs["source"]
        try:
            source = get_source(source_name)
        except KeyError:
            return ToolResult(success=False, error=f"Unknown stock source: {source_name}")
        if not source.is_available():
            return ToolResult(success=False, error=f"Stock source is unavailable: {source_name}")

        filters = SearchFilters(
            kind=inputs.get("kind", "video"),
            min_duration=inputs.get("min_duration"),
            max_duration=inputs.get("max_duration"),
            orientation=inputs.get("orientation"),
            min_width=inputs.get("min_width"),
            per_page=min(max(int(inputs.get("per_page", 8)), 1), 20),
            page=max(int(inputs.get("page", 1)), 1),
        )
        try:
            candidates = source.search(inputs["query"], filters)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            category = "rate_limited" if status == 429 else "provider_error"
            # Do not return exception text: requests exceptions may contain
            # complete request URLs or credential-bearing query parameters.
            return ToolResult(
                success=False,
                error=f"{source_name} search failed ({category}{f', HTTP {status}' if status else ''}; {type(exc).__name__}).",
                data={"source": source_name, "query": inputs["query"], "error_category": category, "http_status": status},
                duration_seconds=round(time.perf_counter() - started, 3),
            )

        rows = [_candidate_metadata(source, candidate, inputs["query"]) for candidate in candidates]
        return ToolResult(
            success=True,
            data={
                "source": source_name,
                "provider": getattr(source, "provider", source_name),
                "query": inputs["query"],
                "kind": filters.kind,
                "candidate_count": len(rows),
                "candidates": rows,
                "downloaded": False,
            },
            cost_usd=0.0,
            duration_seconds=round(time.perf_counter() - started, 3),
        )


def _candidate_metadata(source: Any, candidate: Any, query: str) -> dict[str, Any]:
    """Serialize adapter metadata without manufacturing unavailable fields."""
    extra = dict(getattr(candidate, "extra", {}) or {})
    tags = getattr(candidate, "source_tags", "") or ""
    description = extra.get("description") or extra.get("title") or None
    width, height = int(getattr(candidate, "width", 0) or 0), int(getattr(candidate, "height", 0) or 0)
    duration = float(getattr(candidate, "duration", 0) or 0)
    return {
        "provider": getattr(source, "provider", getattr(candidate, "source", "")),
        "source": candidate.source,
        "asset_id": candidate.clip_id,
        "source_asset_id": candidate.source_id,
        "description": description,
        "tags": extra.get("tags") if isinstance(extra.get("tags"), list) else [part.strip() for part in tags.replace(",", " ").split() if part.strip()],
        "source_tags": tags,
        "duration_seconds": duration if duration > 0 else None,
        "resolution": {"width": width, "height": height} if width and height else None,
        "orientation": _orientation(width, height),
        "download_url": candidate.download_url or None,
        "source_url": candidate.source_url or None,
        "thumbnail_url": candidate.thumbnail_url or None,
        "license": candidate.license or None,
        "creator": candidate.creator or None,
        "kind": candidate.kind,
        "query": query,
        "extra": extra,
        "metadata_basis": "provider search response; visual content not verified",
    }


def _orientation(width: int, height: int) -> str | None:
    if width <= 0 or height <= 0:
        return None
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"
