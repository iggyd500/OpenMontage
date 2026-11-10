# Configured Text Generation

OpenMontage exposes one registry-discovered `text_generation` BaseTool. It
routes through the provider explicitly selected in `config.yaml`:

- `llm.provider: gemini` uses the shared Google GenAI client and configured
  `llm.model` (default `gemini-3.5-flash-lite`).
- `llm.provider: ollama` uses the configured local endpoint and `llm.model`.
- A provider failure is returned to the agent; **there is no automatic provider
  fallback**. Ask before changing providers.

For local use, set `llm.provider: ollama` and `llm.model` to a model tag already
installed in Ollama (for example, `qwen3:8b`). The endpoint defaults to
`http://localhost:11434`; configure `llm.ollama_base_url` or set
`OLLAMA_BASE_URL` to override it.

The tool accepts `task`, optional `instructions`, and optional `source_context`.
Its successful result has the common shape:

Invoke the already-discovered registry tool for each discrete drafting task;
never instantiate a provider-specific client in a pipeline director:

```python
from tools.tool_registry import registry

registry.discover()
tool = registry.get("text_generation")
if tool is None:
    raise RuntimeError("text_generation was not discovered")
result = tool.execute({
    "task": "Draft a storyboard from the approved script",
    "instructions": "Return scene beats with timestamps and visual direction.",
    "source_context": {"script": approved_script, "playbook": active_playbook},
})
if not result.success:
    # Surface the provider error and stop this draft; do not switch providers.
    raise RuntimeError(result.error or "Text generation failed")
draft_text = result.data["text"]
```

The preflight step has already discovered the registry and checked availability.
The selected tool instance reads `config.yaml`, so the call does not pass an
alternate provider or model.

```json
{
  "text": "draft text",
  "provider": "gemini",
  "model": "gemini-3.5-flash-lite",
  "metadata": {}
}
```

Use it as a drafting assistant at these production stages:

| Stage | Good use |
|---|---|
| Research | Synthesize claims and themes from retrieved, cited source material. The tool does not perform web research or verify facts. |
| Script | Draft narration to the approved duration, audience, tone, and source constraints. |
| Scene plan / storyboard | Draft scene beats and visual descriptions from the approved script and playbook. |
| Assets | Refine image/video/diagram prompts from approved scene descriptions. Do not ask this text tool to generate media. |

The stage director/Codex agent must inspect the model output, verify claims
against supplied sources, enforce the relevant JSON schema and creative brief,
write the canonical artifact, and follow review/checkpoint gates. Never let the
tool write artifacts or checkpoints directly. Include only source material that
may be sent to the configured provider; Gemini requests leave the machine.

Routine registry discovery and preflight are offline: status checks inspect
configuration and credentials only. Ollama endpoint reachability and model
availability are tested only when a generation request is explicitly executed.
