"""Offline contract and provider tests for configured text generation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from lib.config_model import LLMConfig
from lib.pipeline_loader import list_pipelines, load_pipeline
from tools.text_generation import TextGeneration
from tools.tool_registry import ToolRegistry


class FakeResponse:
    status_code = 200
    ok = True

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def _gemini_tool(monkeypatch, response=None, error=None):
    import google.genai.types as genai_types
    import tools.google_credentials as google_credentials

    monkeypatch.setattr(
        google_credentials, "has_google_credentials", lambda: True
    )
    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=(
                lambda **kwargs: (_ for _ in ()).throw(error)
                if error
                else response
            )
        )
    )
    monkeypatch.setattr(google_credentials, "get_genai_client", lambda: fake_client)
    monkeypatch.setattr(genai_types, "GenerateContentConfig", lambda **kwargs: kwargs)
    return TextGeneration(LLMConfig())


def test_default_config_selects_gemini_and_expected_model():
    config = LLMConfig()
    assert config.provider == "gemini"
    assert config.model == "gemini-3.5-flash-lite"


@pytest.mark.parametrize(
    "values",
    [
        {"provider": "anthropic"},
        {"provider": "gemini", "model": "qwen3:8b"},
        {"provider": "ollama", "model": "gemini-3.5-flash-lite"},
        {"temperature": 2.1},
        {"max_tokens": 0},
        {"request_timeout_seconds": 0},
        {"ollama_base_url": "file:///tmp/ollama"},
        {"ollama_base_url": "http://user:password@localhost:11434"},
        {"model": "  "},
    ],
)
def test_invalid_llm_config_is_rejected(values):
    with pytest.raises(ValidationError):
        LLMConfig(**values)


def test_gemini_generation_uses_configured_model_and_returns_common_shape(monkeypatch):
    response = SimpleNamespace(
        text="A concise script.",
        candidates=[SimpleNamespace(finish_reason="STOP")],
        usage_metadata=SimpleNamespace(
            prompt_token_count=12, candidates_token_count=5, total_token_count=17
        ),
    )
    tool = _gemini_tool(monkeypatch, response=response)
    result = tool.execute(
        {
            "task": "Write a script",
            "instructions": "Use an engaging voice.",
            "source_context": {"fact": "provided"},
        }
    )
    assert result.success
    assert result.data["text"] == "A concise script."
    assert result.data["provider"] == "gemini"
    assert result.data["model"] == "gemini-3.5-flash-lite"
    assert result.data["metadata"]["total_token_count"] == 17
    assert result.cost_usd is None


def test_gemini_missing_credentials_is_clear_and_does_not_call_client(monkeypatch):
    import tools.google_credentials as google_credentials

    monkeypatch.setattr(google_credentials, "has_google_credentials", lambda: False)
    tool = TextGeneration(LLMConfig())
    result = tool.execute({"task": "draft"})
    assert not result.success
    assert "authentication" in result.error.lower()
    assert "GOOGLE_API_KEY" in result.error


def test_gemini_provider_error_is_categorized_and_secret_redacted(monkeypatch):
    class FakeAPIError(Exception):
        code = 503
        status = "UNAVAILABLE"
        message = "transient failure key=not-a-real-secret"

    monkeypatch.setenv("GOOGLE_API_KEY", "not-a-real-secret")
    tool = _gemini_tool(monkeypatch, error=FakeAPIError())
    result = tool.execute({"task": "draft"})
    assert not result.success
    assert "server_unavailable" in result.error
    assert "not-a-real-secret" not in result.error
    assert "HTTP 503" in result.error


def test_ollama_generation_uses_configured_endpoint_and_common_shape(monkeypatch):
    import tools.text_generation as module

    seen = {}

    def fake_post(url, **kwargs):
        seen["url"] = url
        seen["kwargs"] = kwargs
        return FakeResponse(
            {
                "message": {"content": "Local draft."},
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 4,
                "total_duration": 1234,
            }
        )

    monkeypatch.setattr(module.requests, "post", fake_post)
    config = LLMConfig(
        provider="ollama",
        model="qwen3:8b",
        ollama_base_url="http://127.0.0.1:11435/",
    )
    result = TextGeneration(config).execute(
        {"task": "draft", "instructions": "Be concise", "source_context": "facts"}
    )
    assert result.success
    assert result.data["text"] == "Local draft."
    assert result.data["provider"] == "ollama"
    assert result.data["model"] == "qwen3:8b"
    assert result.cost_usd == 0.0
    assert seen["url"] == "http://127.0.0.1:11435/api/chat"
    assert seen["kwargs"]["json"]["messages"][0]["content"] == "Be concise"


def test_ollama_unavailable_server_is_reported_without_fallback(monkeypatch):
    import requests
    import tools.text_generation as module

    def fail_post(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(module.requests, "post", fail_post)
    config = LLMConfig(provider="ollama", model="qwen3:8b")
    result = TextGeneration(config).execute({"task": "draft"})
    assert not result.success
    assert result.data["provider"] == "ollama"
    assert result.data["category"] == "server_unavailable"
    assert "Gemini" not in result.error


def test_ollama_missing_model_is_reported(monkeypatch):
    import tools.text_generation as module

    missing = FakeResponse({"error": "model not found"})
    missing.status_code = 404
    missing.ok = False
    monkeypatch.setattr(module.requests, "post", lambda *args, **kwargs: missing)
    result = TextGeneration(
        LLMConfig(provider="ollama", model="missing-model")
    ).execute({"task": "draft"})
    assert not result.success
    assert result.data["category"] == "model_unavailable"
    assert "pull that model" in result.error


@pytest.mark.parametrize(
    ("provider", "model", "task"),
    [
        ("gemini", "gemini-3.5-flash-lite", "research brief"),
        ("gemini", "gemini-3.5-flash-lite", "video script"),
        ("ollama", "qwen3:8b", "storyboard"),
        ("ollama", "qwen3:8b", "visual prompts"),
    ],
)
def test_workflow_stage_tasks_share_contract(monkeypatch, provider, model, task):
    import tools.text_generation as module

    monkeypatch.setattr(
        module.requests,
        "post",
        lambda *args, **kwargs: FakeResponse({"message": {"content": "draft"}}),
    )
    if provider == "gemini":
        response = SimpleNamespace(
            text="draft", candidates=[], usage_metadata=None
        )
        tool = _gemini_tool(monkeypatch, response=response)
    else:
        tool = TextGeneration(LLMConfig(provider=provider, model=model))
    result = tool.execute(
        {"task": task, "instructions": "Return structured draft", "source_context": {}}
    )
    assert result.success
    assert set(result.data) == {"text", "provider", "model", "metadata"}
    assert result.data["provider"] == provider


def test_registry_discovers_text_generation_tool():
    discovered = ToolRegistry()
    names = discovered.discover()
    tool = discovered.get("text_generation")
    assert "text_generation" in names
    assert tool is not None
    assert tool.capability == "text_generation"
    assert tool.provider == "gemini"


def test_production_manifests_expose_text_generation_in_draft_stages():
    draft_stages = {"research", "script", "scene_plan", "assets"}
    production_pipelines = set(list_pipelines()) - {"framework-smoke"}
    assert production_pipelines
    for pipeline_name in production_pipelines:
        manifest = load_pipeline(pipeline_name)
        for stage in manifest["stages"]:
            if stage["name"] in draft_stages:
                assert "text_generation" in stage.get("tools_available", []), (
                    pipeline_name,
                    stage["name"],
                )


def test_shared_workflow_instruction_covers_drafting_and_agent_responsibilities():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    guide = (root / "AGENT_GUIDE.md").read_text(encoding="utf-8")
    instructions = (root / "skills/core/text-generation.md").read_text(encoding="utf-8")
    for stage in ("research", "script", "storyboard", "visual prompt"):
        assert stage in guide.lower() or stage in instructions.lower()
    assert "Codex agent remains responsible" in guide
    assert "no automatic provider" in instructions


def test_ollama_preflight_status_does_not_probe_endpoint(monkeypatch):
    import tools.text_generation as module

    def forbidden_probe(*args, **kwargs):
        raise AssertionError("preflight must not make network requests")

    monkeypatch.setattr(module.requests, "post", forbidden_probe)
    tool = TextGeneration(LLMConfig(provider="ollama", model="qwen3:8b"))
    assert tool.get_status().value == "available"
