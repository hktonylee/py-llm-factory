import os

import pip_importer


def get_openai_compatible_client(
    provider: str = None,
    api_key: str = None,
    base_url: str = None,
):
    provider = provider or os.environ.get('LLM_FACTORY__PROVIDER')
    api_key = api_key or os.environ.get('LLM_FACTORY__API_KEY')
    base_url = base_url or os.environ.get('LLM_FACTORY__BASE_URL')

    if provider == 'openai':
        pip_importer.pip_import("openai")
        from openai import OpenAI
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    elif provider == 'groq':
        # ai_base_url = 'https://api.groq.com/openai/v1'
        pip_importer.pip_import("groq")
        from groq import Groq
        return Groq(api_key=api_key)
    else:
        raise ValueError('Unsupported LLM provider: ' + provider)


def get_suggested_nano_model(provider: str = None):
    provider = provider or os.environ.get('LLM_FACTORY__PROVIDER')

    model = os.environ.get('LLM_FACTORY__SUGGESTED_NANO_MODEL')
    if model:
        return model

    if provider == 'openai':
        return 'gpt-5-nano-2025-08-07'
    elif provider == 'groq':
        return 'openai/gpt-oss-20b'
    else:
        raise ValueError('Unsupported LLM provider: ' + provider)


def get_suggested_mini_model(provider: str = None):
    provider = provider or os.environ.get('LLM_FACTORY__PROVIDER')
    model = os.environ.get('LLM_FACTORY__SUGGESTED_MINI_MODEL')
    if model:
        return model

    if provider == 'openai':
        return 'gpt-5-mini-2025-08-07'
    elif provider == 'groq':
        return 'openai/gpt-oss-120b'
    else:
        raise ValueError('Unsupported LLM provider: ' + provider)
