"""Download one explicitly shortlisted stock candidate via its source adapter."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile, ToolResult,
    ToolRuntime, ToolStability, ToolStatus, ToolTier,
)


class StockCandidateDownload(BaseTool):
    name = "stock_candidate_download"
    version = "0.1.0"
    tier = ToolTier.SOURCE
    capability = "clip_acquisition"
    provider = "openmontage"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.HYBRID

    dependencies = ["python:requests"]
    capabilities = ["selected_stock_asset_download", "stock_asset_provenance"]
    supports = {"existing_stock_sources": True, "prevents_overwrite": True, "project_scoped_output": True}
    best_for = ["downloading only an explicitly selected candidate", "preserving source metadata in the asset manifest"]
    side_effects = ["downloads one selected media file", "writes output_path"]
    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=128, vram_mb=0, disk_mb=500, network_required=True)

    input_schema = {
        "type": "object",
        "required": ["candidate", "output_path"],
        "properties": {"candidate": {"type": "object"}, "output_path": {"type": "string"}},
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
        from tools.video.stock_sources import Candidate, get_source

        data = inputs["candidate"]
        output = Path(inputs["output_path"]).resolve()
        projects_root = _projects_root()
        if not output.is_relative_to(projects_root):
            return ToolResult(success=False, error="Selected stock assets must be saved under the OpenMontage projects directory.")
        if output.exists():
            return ToolResult(success=False, error=f"Refusing to overwrite existing asset: {output}")
        download_url = str(data.get("download_url") or "")
        if urlparse(download_url).scheme != "https":
            return ToolResult(success=False, error="Candidate has no valid HTTPS download URL.")
        required = ("source", "source_asset_id", "source_url", "kind")
        if any(not data.get(key) for key in required):
            return ToolResult(success=False, error="Candidate is missing required source provenance fields.")

        try:
            source = get_source(str(data["source"]))
            if not source.is_available():
                return ToolResult(success=False, error=f"Selected source is unavailable: {data['source']}")
            resolution = data.get("resolution") or {}
            candidate = Candidate(
                source=str(data["source"]), source_id=str(data["source_asset_id"]),
                source_url=str(data["source_url"]), download_url=download_url,
                kind=str(data["kind"]), width=int(resolution.get("width") or 0),
                height=int(resolution.get("height") or 0),
                duration=float(data.get("duration_seconds") or 0),
                creator=str(data.get("creator") or ""), license=str(data.get("license") or ""),
                source_tags=str(data.get("source_tags") or ""),
                thumbnail_url=str(data.get("thumbnail_url") or ""),
                extra=dict(data.get("extra") or {}),
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            saved = Path(source.download(candidate, output)).resolve()
            if not saved.is_relative_to(projects_root) or not saved.is_file() or saved.stat().st_size == 0:
                return ToolResult(success=False, error="Stock download did not produce a valid project-local file.")
            return ToolResult(
                success=True,
                data={
                    "asset_id": candidate.clip_id,
                    "provider": getattr(source, "provider", candidate.source),
                    "source": candidate.source,
                    "source_url": candidate.source_url,
                    "path": str(saved),
                    "file_size_bytes": saved.stat().st_size,
                    "duration_seconds": candidate.duration or None,
                    "resolution": {"width": candidate.width, "height": candidate.height} if candidate.width and candidate.height else None,
                    "license": candidate.license or None,
                    "creator": candidate.creator or None,
                },
                artifacts=[str(saved)],
                cost_usd=0.0,
                duration_seconds=round(time.perf_counter() - started, 3),
            )
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            category = "rate_limited" if status == 429 else "download_error"
            # Requests errors can embed the credential-bearing CDN URL.
            return ToolResult(
                success=False,
                error=f"Selected stock asset download failed ({category}{f', HTTP {status}' if status else ''}; {type(exc).__name__}).",
                data={"error_category": category, "http_status": status},
                duration_seconds=round(time.perf_counter() - started, 3),
            )


def _projects_root() -> Path:
    return (Path(__file__).resolve().parents[2] / "projects").resolve()
