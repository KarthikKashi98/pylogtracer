"""
llm_factory.py
===============
Provider-agnostic LangChain LLM factory.

Supports:
  - OpenAI      (gpt-4o, gpt-3.5-turbo, etc.)
  - Anthropic   (claude-3-5-sonnet, etc.)
  - Ollama      (qwen2.5:7b, llama3, etc. — local)
  - Custom      (any OpenAI-compatible base URL — own API, vLLM, LM Studio, etc.)

Config priority (highest → lowest):
  1. Dict passed at init        — production use
  2. Environment variables      — testing / CI

Environment variables:
  LLM_PROVIDER        = openai | anthropic | ollama | custom
  LLM_MODEL           = model name
  LLM_API_KEY         = api key (not needed for ollama)
  LLM_BASE_URL        = custom base URL (for ollama or custom provider)
  LLM_TEMPERATURE     = float, default 0.0
  LLM_MAX_TOKENS      = int,   default 1024

Usage:
    # From env vars (testing)
    factory = LLMFactory()
    llm     = factory.get_llm()

    # From config dict (production / package init)
    factory = LLMFactory({
        "provider":    "openai",
        "model":       "gpt-4o",
        "api_key":     "sk-...",
        "temperature": 0.0,
    })
    llm            = factory.get_llm()           # BaseChatModel for chains
    structured_llm = factory.get_structured_llm(MySchema)  # JSON output
"""

import os
from typing import Optional, Dict, Any, TYPE_CHECKING
from dotenv import load_dotenv

# LangChain base types — imported for type checking only
# At runtime we use Any to avoid hard dependency crashes
if TYPE_CHECKING:
    pass

try:
    from langchain_core.language_models import BaseChatModel as _BaseChatModel
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel as _BaseModel
    LANGCHAIN_AVAILABLE = True
except ImportError:
    _BaseChatModel = None  # type: ignore[assignment,misc]
    JsonOutputParser = None  # type: ignore[assignment,misc]
    ChatPromptTemplate = None  # type: ignore[assignment,misc]
    _BaseModel = None  # type: ignore[assignment,misc]
    LANGCHAIN_AVAILABLE = False
# Load .env only if it exists — config dict takes priority
if os.path.exists('.env'):
    load_dotenv()


# ── Provider constants ────────────────────────────────────────────
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OLLAMA = "ollama"
PROVIDER_CUSTOM = "custom"

SUPPORTED_PROVIDERS = {PROVIDER_OPENAI, PROVIDER_ANTHROPIC,
                       PROVIDER_OLLAMA, PROVIDER_CUSTOM}

# Default models per provider
DEFAULT_MODELS = {
    PROVIDER_OPENAI: "gpt-4o-mini",
    PROVIDER_ANTHROPIC: "claude-3-5-haiku-20241022",
    PROVIDER_OLLAMA: "qwen2.5:7b",
    PROVIDER_CUSTOM: "default",
}


class LLMFactory:
    """
    Creates LangChain LLM instances for any supported provider.

    Args:
        config: Optional dict to override env vars.
                Keys: provider, model, api_key, base_url,
                      temperature, max_tokens
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._llm: Optional[Any] = None  # cached instance

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def get_llm(self) -> Any:
        """
        Return a LangChain BaseChatModel for the configured provider.
        Result is cached — same instance returned on repeated calls.
        """
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm

    def get_structured_llm(self, schema: Any) -> Any:
        """
        Return an LLM that outputs structured JSON matching the schema.
        Used by ErrorTypeClassifier for batch classification.

        Args:
            schema: Pydantic BaseModel class defining the output shape.

        Returns:
            Runnable chain: prompt | llm.with_structured_output(schema)
        """
        llm = self.get_llm()
        return llm.with_structured_output(schema)

    def get_provider(self) -> str:
        return self._resolve("provider", "LLM_PROVIDER", PROVIDER_OLLAMA)

    def get_model(self) -> str:
        provider = self.get_provider()
        return self._resolve("model", "LLM_MODEL", DEFAULT_MODELS[provider])

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — builder
    # ─────────────────────────────────────────────────────────────

    def _build_llm(self) -> Any:
        provider = self.get_provider()
        model = self.get_model()
        api_key = self._resolve("api_key", "LLM_API_KEY", None)
        base_url = self._resolve("base_url", "LLM_BASE_URL", None)
        temperature = float(self._resolve("temperature", "LLM_TEMPERATURE", 0.0))
        max_tokens = int(self._resolve("max_tokens", "LLM_MAX_TOKENS", 1024))

        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Choose from: {SUPPORTED_PROVIDERS}"
            )

        print(f"  [LLMFactory] provider={provider} | model={model}")

        if provider == PROVIDER_OPENAI:
            return self._build_openai(model, api_key, base_url, temperature,
                                      max_tokens)

        elif provider == PROVIDER_ANTHROPIC:
            return self._build_anthropic(model, api_key, temperature, max_tokens)

        elif provider == PROVIDER_OLLAMA:
            return self._build_ollama(model, base_url, temperature)

        elif provider == PROVIDER_CUSTOM:
            return self._build_custom(model, api_key, base_url, temperature,
                                      max_tokens)

    def _build_openai(self, model, api_key, base_url, temperature, max_tokens):
        from langchain_openai import ChatOpenAI
        kwargs = dict(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    def _build_anthropic(self, model, api_key, temperature, max_tokens):
        from langchain_anthropic import ChatAnthropic
        kwargs = dict(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if api_key:
            kwargs["anthropic_api_key"] = api_key
        return ChatAnthropic(**kwargs)

    def _build_ollama(self, model, base_url, temperature):
        from langchain_ollama import ChatOllama
        kwargs = dict(
            model=model,
            temperature=temperature,
        )
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)

    def _build_custom(self, model, api_key, base_url, temperature, max_tokens):
        """
        Any OpenAI-compatible API (vLLM, LM Studio, Together, Groq, etc.)
        Uses ChatOpenAI with a custom base_url.
        """
        from langchain_openai import ChatOpenAI
        if not base_url:
            raise ValueError(
                "provider='custom' requires base_url. "
                "Set LLM_BASE_URL env var or pass base_url in config."
            )
        kwargs = dict(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — config resolution
    # ─────────────────────────────────────────────────────────────

    def _resolve(self, config_key: str, env_key: str, default: Any) -> Any:
        """
        Resolve a config value.
        Priority: config dict → env var → default
        """
        if config_key in self._config and self._config[config_key] is not None:
            return self._config[config_key]
        env_val = os.getenv(env_key)
        if env_val is not None:
            return env_val
        return default
