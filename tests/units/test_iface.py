import pytest
from unittest.mock import patch,AsyncMock,MagicMock
import pandas as pd
import json, os
from semantic_kernel import Kernel
from semantic_kernel.functions.kernel_function import KernelFunction
from semantic_kernel.prompt_template.input_variable import InputVariable
from maif_corpus.setup_interface import (
    process_docs,
    build_kernel_RAG,
    build_kernel,
    create_memory,
    send_query_RAG,
    fill_automatically,
    new_kernel,
)
fixture_AO = json.load(open(os.path.join(os.getcwd(), "tests/fixtures/Dict_AO.json")))

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
AZURE_OPENAI_DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")
DOCKER_MODE = os.getenv('DOCKER_MODE')

VARS = [
    InputVariable(name="question", description="The user input", is_required=True),
    InputVariable(name="context", description=" Retriever", is_required=False),
]

@pytest.mark.parametrize(
    "UI_Corpus, Api_Corpus",
    [
        (fixture_AO, None),
        (None, fixture_AO),
    ],
)
def test_process_docs(UI_Corpus, Api_Corpus):
    result, status = process_docs(UI_Corpus, Api_Corpus)
    assert isinstance(result, dict)
    assert status == "Prêt à apprendre 🟢"

# Test for build_kernel_RAG function
def test_build_kernel_RAG(env_setup):
    kernel = build_kernel()
    RAG = build_kernel_RAG(kernel)
    assert isinstance(RAG, KernelFunction)


# Test for create_memory function
@pytest.mark.asyncio  
async def test_create_memory(env_setup):  
    # Mock the Kernel object  
    with patch('maif_corpus.setup_interface.build_kernel', autospec=True) as MockKernel:  
        # Configure the mock Kernel instance  
        mock_kernel = MockKernel.return_value
        mock_kernel.get_function.return_value.invoke = AsyncMock()

        # Define the test inputs  
        texts = {"doc1": {"full_text": {"page1": ["chunk1", "chunk2"]}}}  
        collection_name = "test_collection"  
  
        # Call the function with the test inputs  
        result = await create_memory(texts, collection_name, mock_kernel)  
  
        # Check that the function returned the expected result  
        assert result == "Prêt à vous répondre 🟢"  
  
        # Check that the kernel's save function was called with the correct arguments  
        mock_kernel.get_function.assert_called_with("TextMemoryPlugin", "save")  
        mock_kernel.get_function.return_value.invoke.assert_called_with(  
            mock_kernel, text="chunk2", key="doc1-page1-1", collection=collection_name  
        )  

@pytest.mark.asyncio  
async def test_send_query_RAG(env_setup):
    with patch('maif_corpus.setup_interface.build_kernel', autospec=True) as MockBuildKernel:  
        with patch('maif_corpus.setup_interface.build_kernel_RAG', autospec=True) as MockBuildKernelRAG:  
            mock_kernel_build = MockBuildKernel.return_value  
            mock_kernel = MockBuildKernelRAG.return_value
            mock_kernel.get_function = MagicMock(return_value=AsyncMock())
            mock_kernel.get_function.return_value.__call__ = AsyncMock()
            mock_kernel.__dict__.update(mock_kernel_build.__dict__)
            with patch('maif_corpus.setup_interface.create_memory', new_callable=AsyncMock) as mock_create_memory:  
                query = "What is Python ?"  
                collection_name = "test_collection"  
                mock_kernel.get_function= MagicMock(return_value=AsyncMock())
                mock_kernel.get_function.return_value.__call__ = AsyncMock()
                result = await send_query_RAG(query, collection_name, mock_kernel)
                assert isinstance(result, MagicMock) 

@pytest.mark.asyncio
async def test_fill_automatically():
    df_UI = pd.DataFrame({"Question": ["What is Python?", "Who created Python?"]})
    NomCorpus = "test_collection"
    df_API = None
    result = await fill_automatically(df_UI, NomCorpus, df_API)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], dict)

def test_new_kernel(env_setup):
    result = new_kernel()
    assert isinstance(result, Kernel)
