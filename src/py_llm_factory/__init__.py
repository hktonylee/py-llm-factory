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
