"""VLM backend adapters for the node labeler.

All adapters speak a common interface: ``complete(system, content, max_tokens)``
where ``content`` is a list of Anthropic-style blocks::

    {"type": "text", "text": "..."}
    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}

Use ``make_vlm_client(model)`` to auto-select an adapter by model name:

    claude-*          → AnthropicVLMClient   (pip install anthropic)
    gemini-3.5-flash  → GeminiVLMClient      (pip install google-genai)  ← default
    gpt-* / o1-*      → OpenAIVLMClient      (pip install openai)
    ollama:<model>    → OllamaVLMClient      (pip install openai; Ollama running locally)
    openai:<model>@<base_url>  → OpenAIVLMClient with custom base URL
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import Any


class VLMClient(ABC):
    """Minimal interface shared by all VLM backends.

    Every subclass accumulates token usage across all its ``complete()``
    calls into ``self.usage`` (``{"input_tokens", "output_tokens",
    "total_tokens"}``) — useful for e.g. reporting the cost of a VLM
    labeling run or a chat session.
    """

    def __init__(self) -> None:
        self.usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def _accumulate_usage(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.usage["input_tokens"] += input_tokens or 0
        self.usage["output_tokens"] += output_tokens or 0
        self.usage["total_tokens"] += (input_tokens or 0) + (output_tokens or 0)

    @abstractmethod
    def complete(
        self,
        system: str,
        content: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> str:
        """Send a request and return the response text.

        Args:
            system: System prompt.
            content: List of Anthropic-style content blocks (text and/or image).
            max_tokens: Maximum tokens to generate, or None for the provider's
                own default. Prefer None: every task here asks for a short JSON
                object and stops on its own, so a cap can only truncate. It is
                especially dangerous on a reasoning model, which spends this
                budget on internal reasoning first and returns an empty or
                half-finished answer once it runs out.
        """


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


# The Messages API requires max_tokens, so None can't mean "omit it" here the
# way it does for the other two. Set high enough to be a runaway guard rather
# than a real limit: these prompts ask for a few hundred tokens of JSON.
_ANTHROPIC_FALLBACK_MAX_TOKENS = 16384


class AnthropicVLMClient(VLMClient):
    """Wraps the official Anthropic SDK. Requires ``pip install anthropic``."""

    def __init__(self, model: str = "claude-opus-4-8", **kwargs: Any) -> None:
        super().__init__()
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self.model = model
        self._client = anthropic.Anthropic(**kwargs)

    def complete(self, system: str, content: list[dict], max_tokens: int | None = None) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens or _ANTHROPIC_FALLBACK_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        if resp.usage:
            self._accumulate_usage(resp.usage.input_tokens, resp.usage.output_tokens)
        return resp.content[0].text


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------


class GeminiVLMClient(VLMClient):
    """Wraps the Google Gen AI SDK. Requires ``pip install google-genai``.

    Uses Vertex AI automatically when ``GOOGLE_GENAI_USE_VERTEXAI=True`` is set
    (along with ``GOOGLE_CLOUD_PROJECT`` and ``GOOGLE_CLOUD_LOCATION``).
    Otherwise falls back to Google AI Studio with ``GOOGLE_API_KEY``.

    Reasoning is requested at ``thinking_level="high"``: reading a grid of
    robot frames and deciding what manipulation it shows is exactly the kind
    of work that benefits, and the answers are a few hundred tokens either
    way, so the extra reasoning tokens are the whole cost.

    That pairs with ``max_tokens=None``. Gemini counts reasoning against
    ``max_output_tokens`` and spends it before writing any answer, so a cap
    that looks generous can still leave nothing for the response: at
    ``max_output_tokens=256`` gemini-3.5-flash used 246 reasoning tokens and
    emitted six tokens of a half-finished sentence, and a heavier prompt
    returns a candidate whose only parts are thoughts (``.text == ""``).
    """

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        thinking_level: str | None = "high",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        import os

        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError("pip install google-genai")
        self.model = model
        self.thinking_level = thinking_level
        self._genai = genai
        self._types = genai_types
        use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true")
        if use_vertex:
            # ADC + project/location come from env vars; don't pass api_key
            kwargs.pop("api_key", None)
            self._client = genai.Client(vertexai=True, **kwargs)
        else:
            api_key = kwargs.pop("api_key", None)
            self._client = genai.Client(api_key=api_key, **kwargs)

    def _to_parts(self, content: list[dict]) -> list[Any]:
        parts = []
        for block in content:
            if block["type"] == "text":
                parts.append(self._types.Part.from_text(text=block["text"]))
            elif block["type"] == "image":
                src = block["source"]
                if src["type"] == "base64":
                    raw = base64.b64decode(src["data"])
                    parts.append(
                        self._types.Part.from_bytes(
                            data=raw,
                            mime_type=src.get("media_type", "image/jpeg"),
                        )
                    )
        return parts

    def complete(self, system: str, content: list[dict], max_tokens: int | None = None) -> str:
        parts = self._to_parts(content)
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            thinking_config=(
                None
                if self.thinking_level is None
                else self._types.ThinkingConfig(thinking_level=self.thinking_level)
            ),
        )
        resp = self._client.models.generate_content(
            model=self.model,
            contents=parts,
            config=config,
        )
        usage = resp.usage_metadata
        if usage:
            # candidates_token_count excludes reasoning, which is billed as
            # output — without thoughts_token_count the run's reported cost is
            # a fraction of the real one.
            self._accumulate_usage(
                usage.prompt_token_count,
                (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
            )
        # Empty, not just None: a response that spent its budget on reasoning
        # has a candidate whose only parts are thoughts, and `.text` is "".
        if not (resp.text or "").strip():
            reason = None
            if resp.candidates:
                reason = resp.candidates[0].finish_reason
            raise RuntimeError(
                f"Gemini returned no text (finish_reason={reason}). "
                "If MAX_TOKENS, raise or drop max_tokens — internal reasoning "
                "comes out of the same budget and is spent first. "
                "If SAFETY, the prompt was blocked."
            )
        return resp.text


# ---------------------------------------------------------------------------
# OpenAI-compatible (OpenAI, Ollama, vLLM, LM Studio, …)
# ---------------------------------------------------------------------------


class OpenAICompatVLMClient(VLMClient):
    """OpenAI chat-completions client. Works with OpenAI, Ollama, vLLM, LM Studio.

    Requires ``pip install openai``.

    For Ollama running locally::

        OpenAICompatVLMClient(
            model="llava",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self.model = model
        client_kwargs: dict[str, Any] = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        client_kwargs.update(kwargs)
        self._client = OpenAI(**client_kwargs)

    def _to_messages(self, system: str, content: list[dict]) -> list[dict]:
        oai_content: list[dict] = []
        for block in content:
            if block["type"] == "text":
                oai_content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                src = block["source"]
                if src["type"] == "base64":
                    mime = src.get("media_type", "image/jpeg")
                    url = f"data:{mime};base64,{src['data']}"
                    oai_content.append({"type": "image_url", "image_url": {"url": url}})
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": oai_content},
        ]

    def complete(self, system: str, content: list[dict], max_tokens: int | None = None) -> str:
        messages = self._to_messages(system, content)
        # Omitted, not sent as null: the openai SDK serializes an explicit
        # None into the body, and strict OpenAI-compatible servers (vLLM,
        # some LM Studio builds) reject `"max_tokens": null` outright.
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            **({} if max_tokens is None else {"max_tokens": max_tokens}),
        )
        if resp.usage:
            self._accumulate_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_vlm_client(model: str, thinking_level: str | None = None, **kwargs: Any) -> VLMClient:
    """Instantiate the right VLM adapter from a model string.

    Naming conventions::

        claude-opus-4-8          → AnthropicVLMClient
        gemini-3.5-flash         → GeminiVLMClient
        gpt-4o                   → OpenAICompatVLMClient (OpenAI)
        ollama:llava             → OpenAICompatVLMClient (localhost:11434)
        ollama:llava@http://...  → OpenAICompatVLMClient (custom Ollama URL)
        openai:<model>@<url>     → OpenAICompatVLMClient with custom base URL

    ``thinking_level`` is named rather than left to ``**kwargs`` because only
    the Gemini adapter understands it — forwarding it blindly would hand an
    unexpected argument to ``anthropic.Anthropic`` / ``OpenAI``. Asking for it
    on another backend raises instead of being dropped: a reasoning setting
    that quietly does nothing is worse than one that says it can't.
    """
    if thinking_level is not None and not model.startswith("gemini-"):
        raise ValueError(
            f"thinking_level is a Gemini setting; {model!r} does not support it. "
            "Pass thinking_level=None for other backends."
        )

    if model.startswith("claude-"):
        return AnthropicVLMClient(model=model, **kwargs)

    if model.startswith("gemini-"):
        if thinking_level is not None:
            kwargs["thinking_level"] = thinking_level
        return GeminiVLMClient(model=model, **kwargs)

    if model.startswith("ollama:"):
        rest = model[len("ollama:") :]
        if "@" in rest:
            ollama_model, base_url = rest.split("@", 1)
        else:
            ollama_model = rest
            base_url = "http://localhost:11434/v1"
        return OpenAICompatVLMClient(
            model=ollama_model,
            base_url=base_url,
            api_key=kwargs.pop("api_key", "ollama"),
            **kwargs,
        )

    if model.startswith("openai:"):
        rest = model[len("openai:") :]
        if "@" in rest:
            oai_model, base_url = rest.split("@", 1)
        else:
            oai_model, base_url = rest, None
        return OpenAICompatVLMClient(model=oai_model, base_url=base_url, **kwargs)

    # gpt-*, o1-*, o3-*, etc.
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return OpenAICompatVLMClient(model=model, **kwargs)

    raise ValueError(
        f"Cannot infer VLM adapter from model name '{model}'. "
        "Use a prefix: claude-*, gemini-*, ollama:<model>, openai:<model>, gpt-*. "
        "Or instantiate an adapter directly."
    )
