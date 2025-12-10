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
