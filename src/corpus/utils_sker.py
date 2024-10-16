import os
from datetime import datetime
from typing import Any

import semantic_kernel as ker
from dotenv import load_dotenv
from openai import AzureOpenAI
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
from semantic_kernel.functions.kernel_function import KernelFunction
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable

try:
    load_dotenv(override=True)
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_EMBEDDING = os.getenv("AZURE_OPENAI_EMBEDDING")
except KeyError as keye:
    raise KeyError("Environment variable is missing : ") from keye

try:
    kernel: ker.Kernel = None
    DEFAULT_RAG_PROMPT = """
    Tu es un assistant réputé pour sa fiabilité, et sa capacité à répondre précisément aux questions qui lui sont posées.
    Pour cela :

    Si tu as du contexte, répondre à la question posée sur cette base avec un ton concis et factuel.
    Ajoute en bullet point les éléments du contexte qui t'est transmis pour illustrer ta réponse, avec le titre du document et la page.

    Si tu n'as pas de contexte donné, renvoie strictement la mention "Non trouvé dans le corpus"

    ----------------
    Contexte: {{$context}}
    ----------------
    Question: {{$question}}
    ----------------

    Merci beaucoup pour ton aide, très précieuse
    """

    FUNC_NAME = "chatbot_with_memory_context"
    PLUGIN_NAME = "chatPluginWithContext"
    VARS: list[InputVariable] = [
        InputVariable(name="question", description="The user input", is_required=True),
        InputVariable(name="context", description=" Retriever", is_required=False),
    ]
except Exception as e:
    raise AssertionError("Fail to create RAG config") from e
try:
    DEFAULT_CLASSIFY_PROMPT = """
	Tu es un assistant qui reçoit un ensemble de questions filtrées, et ton but est d'indiquer si {}.
	Tu ne dois rien déduire, simplement répondre parmi les options suivantes qui s'excluent entre elles {}

	Voici l'extrait de la liste de question qui sert de base à ta réponse {}

	Response must be a json with key : anwser, value : the answer
    """
    CLASSIF_NAME = "classifFunction"
    CLASSIF_PLUGIN_NAME = "classifPlugin"
    CLASSIF_VARS: list[InputVariable] = [
        InputVariable(
            name="objective", description="A description of what the user wants to classify", is_required=True
        ),
        InputVariable(name="modalities", description=" The possible reuslts of classification", is_required=False),
        InputVariable(name="input", description=" The QnA used to classify", is_required=False),
    ]  # à introduire dans le prompt une foois passé en mode kernel

except Exception as e:
    raise AssertionError("Fail to create Calssif config") from e


def send_prompt(role: str, prompt: str) -> str:
    """
    Sends a prompt to the Azure OpenAI API and returns the API's response.

    Args:
        role (str): The role for the message, either 'system' or 'user'.
        prompt (str): The content of the message to be sent to the API.

    Returns:
        str: The content of the response message from the API.

    This function initializes a client for the Azure OpenAI API, sends a message to the API, and returns the content
    of the API's response. The message sent to the API consists of a role and a prompt.
    """
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION
    )

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_ID,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": role}, {"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def populate_memory(
    memory: SemanticTextMemory, collection_id: str, doc_id: str, text: str, metadata: str
) -> None:
    """
    Asynchronously saves a reference to the provided text and metadata in the specified semantic text memory.

    Args:
        memory (SemanticTextMemory): The semantic text memory object where the reference will be saved.
        collection_id (str): The identifier of the collection where the reference will be saved.
        doc_id (str): The identifier for the reference that will be saved.
        text (str): The text of the reference that will be saved.
        metadata (str): Additional metadata for the reference that will be saved.

    Returns:
        None
    """
    await memory.save_reference(collection=collection_id, id=doc_id, text=text, additional_metadata=metadata)


async def create_tmp_memory(
    collection_id: str,
    Corpus: dict,
    kernel: ker.Kernel,
    embedding_service: AzureTextEmbedding,
    name_plugin_memory: str = "TextMemoryPlugin",
) -> tuple[ker.Kernel, dict, dict]:
    """
    Asynchronously creates a temporary memory, adds it to the kernel, and populates it with data.

    Args:
        collection_id (str): The identifier of the collection where the reference will be saved.
        Corpus (dict): A dictionary containing documents to be added to the memory.
        kernel (ker.Kernel): The kernel where the memory plugin will be added.
        embedding_service (AzureTextEmbedding): The embedding service to generate embeddings for the memory.
        name_plugin_memory (str, optional): The name of the memory plugin to be added to the kernel. Defaults to "TextMemoryPlugin".

    Returns:
        tuple: A tuple containing the updated kernel, a report of the time taken to add each document,
               and a dictionary of failed additions.

    Raises:
        Exception: Any exception raised while adding the plugin to the kernel or populating the memory.
    """
    try:
        memory = SemanticTextMemory(storage=VolatileMemoryStore(), embeddings_generator=embedding_service)

        kernel.add_plugin(TextMemoryPlugin(memory), name_plugin_memory)
    except Exception as e:
        raise e

    report: dict[Any, Any] = {}
    fail: dict[Any, Any] = {}
    i = 0

    for idd, doc in Corpus.items():
        j = 1
        start = datetime.now()
        if doc["full_text"] and len(doc["full_text"]) > 0:
            for split in doc["full_text"]:
                try:
                    await populate_memory(
                        memory, collection_id=collection_id, doc_id=f"{j} - {idd}", text=split, metadata=idd
                    )
                except BaseException as e:
                    fail[f"{j} - {idd}"] = e
                j += 1
        time = datetime.now() - start
        report[i] = {}
        report[i]["time"] = time.microseconds / 100000
        i += 1

    return kernel, report, fail


def create_template_cfg(
    kernel: ker.Kernel, service: str, variables_prompt: list[InputVariable], prompt: str = DEFAULT_RAG_PROMPT
) -> PromptTemplateConfig:
    """
    Creates a configuration for a prompt template.
    Args:
        kernel (ker.Kernel): The kernel to get the service from.
        service (str): The name of the service to instantiate execution settings from.
        variables_prompt (list[InputVariable]): The input variables for the prompt template.
        prompt (str, optional): The prompt for the template. Defaults to DEFAULT_RAG_PROMPT.

    Returns:
        PromptTemplateConfig: The configuration for the prompt template.
    """
    execution_config = kernel.get_service(service).instantiate_prompt_execution_settings(
        service_id=service, max_tokens=500, temperature=0, seed=42
    )
    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        name="chat",
        template_format="semantic-kernel",
        input_variables=variables_prompt,
        execution_settings=execution_config,
    )
    return prompt_template_config


def attach_cfg_func(
    kernel: ker.Kernel, function_name: str, plugin_name: str, prompt_template_config: str
) -> KernelFunction:
    """
    Attaches a function to the kernel.

    Args:
        kernel (ker.Kernel): The kernel to attach the function to.
        function_name (str): The name of the function to attach.
        plugin_name (str): The name of the plugin the function belongs to.
        prompt_template_config (str): The configuration for the prompt template of the function.

    Returns:
        KernelFunction: The kernel with the attached function.
    """
    kernel_with_func = kernel.add_function(
        function_name=function_name,
        plugin_name=plugin_name,
        prompt_template_config=prompt_template_config,
    )
    return kernel_with_func
