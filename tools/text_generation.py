"""Config-selected Gemini/Ollama text generation behind one BaseTool contract."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
from typing import Any

import requests

from lib.config_model import LLMConfig, OpenMontageConfig
from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from tools.provider_pricing import PriceQuoteRequired


class TextGeneration(BaseTool):
    """Generate text with the explicitly configured Gemini or Ollama provider.

    This is a single registry tool, not a selector: ``config.llm.provider``
    determines exactly one backend, and a failure never switches providers.
    """

    name = "text_generation"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "text_generation"
    provider = "gemini"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API
    dependencies = ["python:requests"]
    install_instructions = (
        "For Gemini, configure GOOGLE_API_KEY in .env. For Ollama, install and "
        "start Ollama, then set llm.provider/model in config.yaml."
    )
    capabilities = ["research_synthesis", "scriptwriting", "storyboarding", "visual_prompting"]
    supports = {
        "provider_selection": "config.llm.provider",
        "context_input": True,
        "offline_provider": "ollama",
        "automatic_fallback": False,
    }
    best_for = [
        "drafting research, scripts, storyboards and visual prompts",
        "Gemini cloud generation or explicitly selected local Ollama generation",
    ]
    not_good_for = [
        "web research without source material or a separate search tool",
        "image, video, speech or music generation",
    ]
    input_schema = {
        "type": "object",
        "required": ["task"],
        "properties": {
            "task": {"type": "string", "minLength": 1},
            "instructions": {"type": "string"},
            "source_context": {
                "description": "Source text or structured evidence for the task.",
                "oneOf": [
                    {"type": "string"},
                    {"type": "object"},
                    {"type": "array"},
                ],
            },
        },
    }
    output_schema = {
        "type": "object",
        "required": ["text", "provider", "model", "metadata"],
        "properties": {
            "text": {"type": "string"},
            "provider": {"type": "string", "enum": ["gemini", "ollama"]},
            "model": {"type": "string"},
            "metadata": {"type": "object"},
        },
    }
    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=0, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=0)
    side_effects = ["sends text task/context to the explicitly configured provider"]
    user_visible_verification = ["Review generated text against sources and stage schema"]

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or OpenMontageConfig.load().llm
        self.provider = self.config.provider
        self.model = self.config.model
        self.runtime = ToolRuntime.LOCAL if self.provider == "ollama" else ToolRuntime.API
        self.resource_profile = ResourceProfile(
            cpu_cores=1,
            ram_mb=256 if self.provider == "gemini" else 1024,
            vram_mb=0,
            disk_mb=0,
            network_required=self.provider == "gemini",
        )

    def get_status(self) -> ToolStatus:
        """Report static configuration only; never make a network probe in preflight."""
        if self.provider == "ollama":
            # The endpoint is validated by LLMConfig; reachability is checked only
            # during an explicitly invoked generation request.
            return ToolStatus.AVAILABLE

        from tools.google_credentials import has_google_credentials

        if not has_google_credentials():
            return ToolStatus.UNAVAILABLE
        try:
            google_sdk_available = importlib.util.find_spec("google.genai") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            google_sdk_available = False
        if not google_sdk_available:
            return ToolStatus.UNAVAILABLE
        return ToolStatus.AVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        if self.provider == "ollama":
            return 0.0
        # Free-tier eligibility depends on the Google project and its current
        # quota/billing state; do not represent an API call as guaranteed free.
        raise PriceQuoteRequired(
            "Gemini text-generation cost depends on the project's current tier and quota"
        )

    @staticmethod
    def _prompt(inputs: dict[str, Any]) -> str:
        task = str(inputs.get("task", "")).strip()
        if not task:
            raise ValueError("task must be a non-empty string")
        context = inputs.get("source_context")
        if context is None:
            context_text = "(none supplied)"
        elif isinstance(context, str):
            context_text = context
        else:
            context_text = json.dumps(context, ensure_ascii=False, indent=2)
        return f"Task:\n{task}\n\nSource context:\n{context_text}"

    @staticmethod
    def _safe_exception_message(exc: Exception) -> str:
        """Redact known secret environment values from provider messages."""
        message = str(getattr(exc, "message", None) or exc)
        secrets = {
            value
            for key, value in os.environ.items()
            if value
            and any(
                part in key.upper()
                for part in ("_API_KEY", "_TOKEN", "_SECRET", "CREDENTIAL")
            )
        }
        for secret in sorted(secrets, key=len, reverse=True):
            message = message.replace(secret, "[REDACTED]")
        message = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED]", message)
        message = re.sub(
            r"([?&](?:key|api_key)=)[^&\s]+", r"\1[REDACTED]", message, flags=re.I
        )
        message = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", message, flags=re.I)
        return message[:800]

    @staticmethod
    def _error_category(status_code: Any, status: Any, message: str) -> str:
        normalized = str(status or "").upper()
        message_lower = message.lower()
        if status_code in (401, 403) or normalized in {
            "UNAUTHENTICATED",
            "PERMISSION_DENIED",
        }:
            return "authentication"
        if status_code == 429 or normalized in {"RESOURCE_EXHAUSTED", "QUOTA_EXCEEDED"}:
            return "quota"
        if status_code in (404,) or normalized in {"NOT_FOUND", "MODEL_NOT_FOUND"}:
            return "model_unavailable"
        try:
            server_error = status_code is not None and int(status_code) >= 500
        except (TypeError, ValueError):
            server_error = False
        if server_error or normalized in {"UNAVAILABLE", "INTERNAL"}:
            return "server_unavailable"
        if "model" in message_lower and any(
            phrase in message_lower for phrase in ("not found", "not available", "does not exist")
        ):
            return "model_unavailable"
        return "provider_error"

    def _execute_gemini(self, prompt: str, instructions: str) -> ToolResult:
        from tools.google_credentials import get_genai_client, has_google_credentials

        if not has_google_credentials():
            return ToolResult(
                success=False,
                error="Gemini authentication is unavailable: configure GOOGLE_API_KEY or supported Google credentials.",
                cost_usd=None,
                model=self.model,
            )
        try:
            from google.genai import types

            client = get_genai_client()
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=instructions or None,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                ),
            )
            generated = getattr(response, "text", None)
            if not generated:
                return ToolResult(
                    success=False,
                    error="Gemini returned no text.",
                    cost_usd=None,
                    model=self.model,
                )
            usage = getattr(response, "usage_metadata", None)
            candidates = getattr(response, "candidates", None) or []
            metadata = {
                "finish_reason": self._enum_value(candidates[0] if candidates else None, "finish_reason"),
                "prompt_token_count": getattr(usage, "prompt_token_count", None),
                "candidates_token_count": getattr(usage, "candidates_token_count", None),
                "total_token_count": getattr(usage, "total_token_count", None),
                "cost_status": "tier_dependent_unquoted",
            }
            return ToolResult(
                success=True,
                data={
                    "text": generated,
                    "provider": "gemini",
                    "model": self.model,
                    "metadata": metadata,
                },
                cost_usd=None,
                model=self.model,
            )
        except Exception as exc:  # Provider SDK errors vary across SDK versions.
            status_code = getattr(exc, "code", None)
            status = getattr(exc, "status", None)
            message = self._safe_exception_message(exc)
            category = self._error_category(status_code, status, message)
            detail = f"Gemini {category} error"
            if status_code is not None:
                detail += f" (HTTP {status_code})"
            if status:
                detail += f" [{status}]"
            if message:
                detail += f": {message}"
            return ToolResult(
                success=False,
                error=detail,
                cost_usd=None,
                model=self.model,
                data={"provider": "gemini", "model": self.model, "category": category},
            )

    @staticmethod
    def _enum_value(candidate: Any, attribute: str) -> str | None:
        value = getattr(candidate, attribute, None) if candidate is not None else None
        return getattr(value, "name", str(value)) if value is not None else None

    def _execute_ollama(self, prompt: str, instructions: str) -> ToolResult:
        base_url = os.environ.get(
            "OLLAMA_BASE_URL", self.config.ollama_base_url
        ).rstrip("/")
        try:
            base_url = LLMConfig(
                provider="ollama", model=self.model, ollama_base_url=base_url
            ).ollama_base_url
        except Exception:
            return ToolResult(
                success=False,
                error="Ollama endpoint configuration is invalid; use an HTTP(S) URL without embedded credentials.",
                cost_usd=0.0,
                model=self.model,
                data={"provider": "ollama", "model": self.model, "category": "invalid_configuration"},
            )
        try:
            response = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": prompt},
                    ],
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    },
                },
                timeout=self.config.request_timeout_seconds,
            )
        except requests.RequestException as exc:
            category = "server_unavailable"
            return ToolResult(
                success=False,
                error=f"Ollama {category}: server unavailable at configured endpoint ({type(exc).__name__}).",
                cost_usd=0.0,
                model=self.model,
                data={"provider": "ollama", "model": self.model, "category": category},
            )

        if response.status_code == 404:
            category = "model_unavailable"
            error = f"Ollama model '{self.model}' is unavailable; pull that model in Ollama first."
            return ToolResult(
                success=False,
                error=error,
                cost_usd=0.0,
                model=self.model,
                data={"provider": "ollama", "model": self.model, "category": category},
            )
        if not response.ok:
            category = "provider_error"
            return ToolResult(
                success=False,
                error=f"Ollama request failed (HTTP {response.status_code}).",
                cost_usd=0.0,
                model=self.model,
                data={"provider": "ollama", "model": self.model, "category": category},
            )
        try:
            payload = response.json()
        except ValueError:
            return ToolResult(
                success=False,
                error="Ollama returned an invalid JSON response.",
                cost_usd=0.0,
                model=self.model,
                data={"provider": "ollama", "model": self.model, "category": "provider_error"},
            )
        generated = (payload.get("message") or {}).get("content")
        if not isinstance(generated, str) or not generated.strip():
            return ToolResult(
                success=False,
                error="Ollama returned no text.",
                cost_usd=0.0,
                model=self.model,
                data={"provider": "ollama", "model": self.model, "category": "provider_error"},
            )
        metadata = {
            "done_reason": payload.get("done_reason"),
            "prompt_token_count": payload.get("prompt_eval_count"),
            "generated_token_count": payload.get("eval_count"),
            "total_duration_ns": payload.get("total_duration"),
            "cost_status": "local_no_api_charge",
        }
        return ToolResult(
            success=True,
            data={"text": generated, "provider": "ollama", "model": self.model, "metadata": metadata},
            cost_usd=0.0,
            model=self.model,
        )

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        """Generate text via only the backend selected in ``config.llm``."""
        started = time.monotonic()
        try:
            prompt = self._prompt(inputs)
        except (TypeError, ValueError) as exc:
            return ToolResult(success=False, error=str(exc), cost_usd=None, model=self.model)
        instructions = str(inputs.get("instructions", "")).strip()
        if self.provider == "gemini":
            result = self._execute_gemini(prompt, instructions)
        elif self.provider == "ollama":
            result = self._execute_ollama(prompt, instructions)
        else:  # Defensive: LLMConfig validates this before tool construction.
            return ToolResult(
                success=False,
                error=f"Unsupported text-generation provider: {self.provider}",
                cost_usd=None,
                model=self.model,
            )
        result.duration_seconds = round(time.monotonic() - started, 3)
        return result
