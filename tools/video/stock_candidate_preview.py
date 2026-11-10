"""Download one small stock thumbnail for visual review before full media."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile, ToolResult,
    ToolRuntime, ToolStability, ToolStatus, ToolTier,
)

MAX_PREVIEW_BYTES = 2 * 1024 * 1024


class StockCandidatePreview(BaseTool):
    name = "stock_candidate_preview"
    version = "0.1.0"
    tier = ToolTier.SOURCE
    capability = "stock_preview"
    provider = "openmontage"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.HYBRID

    dependencies = ["python:requests"]
    capabilities = ["stock_thumbnail_preview"]
    supports = {"size_capped": True, "project_scoped_output": True}
    best_for = ["reviewing shortlisted thumbnails before full video download"]
    side_effects = ["downloads one thumbnail to output_path"]
    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=32, vram_mb=0, disk_mb=2, network_required=True)

    input_schema = {
        "type": "object",
        "required": ["candidate", "output_path"],
        "properties": {"candidate": {"type": "object"}, "output_path": {"type": "string"}},
    }

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        candidate = inputs["candidate"]
        url = str(candidate.get("thumbnail_url") or "")
        if urlparse(url).scheme != "https":
            return ToolResult(success=False, error="Candidate has no valid HTTPS thumbnail URL.")
        output = Path(inputs["output_path"]).resolve()
        projects_root = _projects_root()
        if not output.is_relative_to(projects_root):
            return ToolResult(success=False, error="Stock previews must be saved under the OpenMontage projects directory.")
        if output.exists():
            return ToolResult(success=False, error=f"Refusing to overwrite existing preview: {output}")

        import requests

        try:
            headers = {}
            if candidate.get("source") == "wikimedia":
                # Match the existing WikimediaSource API etiquette. Commons
                # may reject anonymous thumbnail requests with HTTP 403.
                headers["User-Agent"] = "OpenMontageBot/0.1 (https://github.com/calesthio/OpenMontage)"
            with requests.get(url, headers=headers, stream=True, timeout=20) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                if not content_type.startswith("image/"):
                    return ToolResult(success=False, error="Provider preview URL did not return an image.")
                size_header = response.headers.get("Content-Length")
                if size_header and int(size_header) > MAX_PREVIEW_BYTES:
                    return ToolResult(success=False, error="Preview exceeds the 2 MiB size limit.")
                output.parent.mkdir(parents=True, exist_ok=True)
                total = 0
                with output.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=32 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > MAX_PREVIEW_BYTES:
                            stream.close()
                            output.unlink(missing_ok=True)
                            return ToolResult(success=False, error="Preview exceeds the 2 MiB size limit.")
                        stream.write(chunk)
            if total == 0:
                output.unlink(missing_ok=True)
                return ToolResult(success=False, error="Provider returned an empty image preview.")
            return ToolResult(
                success=True,
                data={"asset_id": candidate.get("asset_id"), "path": str(output), "file_size_bytes": total, "content_type": content_type, "max_preview_bytes": MAX_PREVIEW_BYTES},
                artifacts=[str(output)], cost_usd=0.0,
                duration_seconds=round(time.perf_counter() - started, 3),
            )
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            category = "rate_limited" if status == 429 else "preview_error"
            return ToolResult(
                success=False,
                error=f"Stock thumbnail preview failed ({category}{f', HTTP {status}' if status else ''}; {type(exc).__name__}).",
                data={"error_category": category, "http_status": status},
                duration_seconds=round(time.perf_counter() - started, 3),
            )


def _projects_root() -> Path:
    return (Path(__file__).resolve().parents[2] / "projects").resolve()
