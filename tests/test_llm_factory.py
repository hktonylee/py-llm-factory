import os

from openai import OpenAI

import py_llm_factory


def setup_module():
    os.environ["LLM_FACTORY__PROVIDER"] = "openai"
    os.environ["LLM_FACTORY__API_KEY"] = "test"
    os.environ["LLM_FACTORY__BASE_URL"] = "test"


def test_get_openai_compatible_client_from_environment_variables():
    client = py_llm_factory.get_openai_compatible_client()
    assert client is not None
    assert type(client) == OpenAI


def test_get_groq_compatible_client_from_arguments():
    client = py_llm_factory.get_openai_compatible_client(provider="groq")
    assert client is not None


def test_get_groq_compatible_client_from_environment_variables():
    os.environ["LLM_FACTORY__PROVIDER"] = "groq"
    client = py_llm_factory.get_openai_compatible_client()
    assert client is not None


def test_get_preferred_nano_model_from_environment_variables():
    os.environ["LLM_FACTORY__PREFERRED_NANO_MODEL"] = "test112"
    model = py_llm_factory.get_preferred_nano_model()
    assert model == "test112"


def test_get_preferred_mini_model_from_environment_variables():
    os.environ["LLM_FACTORY__PREFERRED_MINI_MODEL"] = "test223"
    model = py_llm_factory.get_preferred_mini_model()
    assert model == "test223"


def test_empty_environ_string_treated_as_none():
    """Empty environment variable strings should be treated as None."""
    os.environ["LLM_FACTORY__PROVIDER"] = ""
    os.environ["LLM_FACTORY__API_KEY"] = ""
    os.environ["LLM_FACTORY__BASE_URL"] = ""

    # Should raise ValueError because provider becomes None (from empty string)
    try:
        py_llm_factory.get_openai_compatible_client()
        assert False, "Expected ValueError for empty provider"
    except ValueError as e:
        assert "Unsupported LLM provider: None" in str(e)


def test_api_key_provider_from_argument():
    """API key provider command should be used to get the API key."""
    os.environ["LLM_FACTORY__PROVIDER"] = "openai"
    os.environ["LLM_FACTORY__API_KEY"] = ""
    os.environ["LLM_FACTORY__BASE_URL"] = "http://test"

    # Use echo command to provide the API key
    client = py_llm_factory.get_openai_compatible_client(
        api_key_provider="echo my-secret-key"
    )
    assert client is not None
    assert client.api_key == "my-secret-key"


def test_api_key_provider_from_environment_variable():
    """API key provider from environment variable should be used to get the API key."""
    os.environ["LLM_FACTORY__PROVIDER"] = "openai"
    os.environ["LLM_FACTORY__API_KEY"] = ""
    os.environ["LLM_FACTORY__BASE_URL"] = "http://test"
    os.environ["LLM_FACTORY__API_KEY_PROVIDER"] = "echo env-secret-key"

    client = py_llm_factory.get_openai_compatible_client()
    assert client is not None
    assert client.api_key == "env-secret-key"

    # Clean up
    del os.environ["LLM_FACTORY__API_KEY_PROVIDER"]


def test_api_key_takes_precedence_over_api_key_provider():
    """Direct api_key should take precedence over api_key_provider."""
    os.environ["LLM_FACTORY__PROVIDER"] = "openai"
    os.environ["LLM_FACTORY__BASE_URL"] = "http://test"

    client = py_llm_factory.get_openai_compatible_client(
        api_key="direct-key", api_key_provider="echo provider-key"
    )
    assert client is not None
    assert client.api_key == "direct-key"
