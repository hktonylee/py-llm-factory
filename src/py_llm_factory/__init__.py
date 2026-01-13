import os
from typing import Optional

import pip_importer


def get_openai_compatible_client(
    provider: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    api_key_provider: Optional[str] = None,
    base_url: Optional[str] = None,
):
    provider = provider or os.environ.get("LLM_FACTORY__PROVIDER") or None
    api_key = api_key or os.environ.get("LLM_FACTORY__API_KEY") or None
    api_key_provider = (
        api_key_provider or os.environ.get("LLM_FACTORY__API_KEY_PROVIDER") or None
    )
    base_url = base_url or os.environ.get("LLM_FACTORY__BASE_URL") or None

    if not api_key and api_key_provider:
        import subprocess

        api_key = (
            subprocess.check_output(api_key_provider, shell=True)
            .decode("utf-8")
            .strip()
        )

    if provider == "openai":
        pip_importer.pip_import("openai")
        from openai import OpenAI

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    elif provider == "groq":
        # ai_base_url = 'https://api.groq.com/openai/v1'
        pip_importer.pip_import("groq")
        from groq import Groq

        return Groq(api_key=api_key)
    else:
        raise ValueError("Unsupported LLM provider: " + str(provider))


def get_preferred_nano_model(provider: Optional[str] = None) -> str:
    provider = provider or os.environ.get("LLM_FACTORY__PROVIDER")

    model = os.environ.get("LLM_FACTORY__PREFERRED_NANO_MODEL")
    if model:
        return model

    if provider == "openai":
        return "gpt-5-nano-2025-08-07"
    elif provider == "groq":
        return "openai/gpt-oss-20b"
    else:
        raise ValueError("Unsupported LLM provider: " + str(provider))


def get_preferred_mini_model(provider: Optional[str] = None) -> str:
    provider = provider or os.environ.get("LLM_FACTORY__PROVIDER")
    model = os.environ.get("LLM_FACTORY__PREFERRED_MINI_MODEL")
    if model:
        return model

    if provider == "openai":
        return "gpt-5-mini-2025-08-07"
    elif provider == "groq":
        return "openai/gpt-oss-120b"
    else:
        raise ValueError("Unsupported LLM provider: " + str(provider))
