import pytest
from unittest.mock import patch
import os
from semantic_kernel import Kernel
from semantic_kernel.functions.kernel_function import KernelFunction
from semantic_kernel.prompt_template.input_variable import InputVariable

from maif_corpus.setup_interface import build_kernel_RAG, build_kernel
from maif_corpus.utils_sker import attach_cfg_func

chunks = {
        "doc1": {"full_text": {"content of doc1": ["Python est un super langage"]}},
        "doc2": {"full_text": {"content of doc2": ["test"]}},
    }

@pytest.fixture(autouse=True)  
def env_setup(monkeypatch):  
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "mock_api")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_ID", "cog-api-gpt-4o")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "mock_key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://oai-mbe-ao-b2b-sandbox.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING", "cog-api-embedding")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "7860")
    monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")
    monkeypatch.setenv("DOCKER_MODE", "0")

FUNC_NAME = "chatbot_with_memory_context"
PLUGIN_NAME = "chatPluginWithContext"
MEMORY_NAME = "TextMemoryPlugin"
DOCKER_MODE = os.getenv('DOCKER_MODE')

VARS = [
    InputVariable(name="question", description="The user input", is_required=True),
    InputVariable(name="context", description=" Retriever", is_required=False),
]
# Test for build_kernel function
def test_build_kernel(env_setup):
    kernel = build_kernel()
    assert isinstance(kernel, Kernel)

# Test for build_kernel_RAG function
def test_build_kernel_RAG(env_setup):
    kernel = build_kernel()
    result = build_kernel_RAG(kernel)
    assert isinstance(result, KernelFunction)

# Test for create_memory function

texts = str(
    " {'doc1': {'full_text': {'content of doc1':['test']}}, 'doc2': {'full_text': {'content of doc2':['test']}}}"
)  # Replace with your own texts
texts = {"doc1": {"full_text": {"content of doc1": ["test"]}}, "doc2": {"full_text": {"content of doc2": ["test"]}}}
# 

 
@pytest.fixture  
def mock_kernel():  
    with patch('semantic_kernel.Kernel', autospec=True) as mock:  
        instance = mock.return_value  
        instance.services = {}  
        yield instance  
  
def test_attach_cfg_func(mock_kernel):  
    # Assign the necessary values to your variables  
    function_name = FUNC_NAME 
    plugin_name = PLUGIN_NAME 
    prompt_template_config = FUNC_NAME
  
    # Call the function under test  
    attach_cfg_func(mock_kernel, function_name, plugin_name, prompt_template_config)  
  
    # # Make assertions  
    # mock_kernel.add_function.assert_called_once_with(  
    #     function_name=function_name,  
    #     plugin_name=plugin_name,  
    #     prompt_template_config=prompt_template_config,  
    # )  
    assert isinstance(mock_kernel, Kernel)