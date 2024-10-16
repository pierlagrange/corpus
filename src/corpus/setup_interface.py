import ast
import asyncio
import base64
import logging
import os

import gradio as gr
import pandas as pd
import semantic_kernel as ker
from dotenv import load_dotenv
from pythonjsonlogger import jsonlogger
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
from semantic_kernel.functions import KernelFunction
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from semantic_kernel.prompt_template.input_variable import InputVariable

from corpus.utils_index import Chunker, Ingestor
from corpus.utils_question import Questions
from corpus.utils_sker import (
    CLASSIF_NAME,
    CLASSIF_PLUGIN_NAME,
    CLASSIF_VARS,
    DEFAULT_CLASSIFY_PROMPT,
    DEFAULT_RAG_PROMPT,
    attach_cfg_func,
    create_template_cfg,
    send_prompt,
)

load_dotenv(override=True)

logger = logging.getLogger()

logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(10)

try:
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_EMBEDDING = os.getenv("AZURE_OPENAI_EMBEDDING")
    GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT"))
    GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME")
    DOCKER_MODE = os.getenv("DOCKER_MODE")
    logger.info(f"Docker Mode {DOCKER_MODE}")
    logger.info("Import env VARS OK ")
except Exception as e:
    logger.info(f"Import env VARS KO - {e}")

try:
    FUNC_NAME = "chatbot_with_memory_context"
    PLUGIN_NAME = "chatPluginWithContext"
    MEMORY_NAME = "TextMemoryPlugin"
    VARS = [
        InputVariable(name="question", description="The user input", is_required=True),
        InputVariable(name="context", description=" Retriever", is_required=False),
    ]
    logger.info("Import Semantic Kernel Default VARS OK ")
except Exception as e:
    logger.info(f"Import Semantic Kernel Default VARS KO - {e}")

try:
    if DOCKER_MODE == "1":
        PATH_SAVE_QUESTION = os.path.join(os.getcwd(), "ref_question/")
    elif DOCKER_MODE == "0":
        PATH_SAVE_QUESTION = os.path.join(os.getcwd(), "src/maif_corpus/ref_question/")
    logger.info("Import Default Forms OK ")
except Exception as e:
    logger.info(f"Import Default Forms KO - {e}")

kernel = ker.Kernel()
kernel_rag = ker.Kernel()


def process_docs(UI_Corpus: gr.Files | None = None, Api_Corpus: gr.JSON | None = None) -> tuple[dict, str] | Exception:
    """
    This function processes documents either from a UI or an API corpus, .

    Args:
        UI_Corpus (gr.Files, optional): The corpus of documents from the UI.
        Api_Corpus (gr.JSON, optional): The corpus of documents from the API.

    Returns:
        Tuple: Contains the chunks output and a status message if successful, or the exception if an error occurred.
    """
    try:
        if (Api_Corpus is not None) and (len(Api_Corpus) > 0):
            doc_to_chunk = {k: base64.b64decode(v) for k, v in Api_Corpus.items()}
            channel = "backend"
            logger.info(
                f"Import documents OK from {channel} - To process {len([x for x in doc_to_chunk.values() if x])}"
            )
        else:
            doc_to_chunk = {str(idd): doc for idd, doc in enumerate(UI_Corpus)}
            channel = "frontend"
            logger.info(
                f"Import documents OK from {channel} - To process {len([x for x in doc_to_chunk.values() if x])}"
            )
        try:
            chunker = Chunker("recursive")
            ingestor = Ingestor(doc_to_chunk, chunker)
            ingestor.guess_format()
            ingestor.process_docs()
            ingestor.create_chunks()
            if len(ingestor.failed_to_process.keys()) > 0:
                logger.warning(
                    f" Partial Chunking {len(ingestor.failed_to_process.keys())} failed to process - {ingestor.failed_to_process}"
                )
            else:
                logger.info("Chunking OK - All files has been processed")

        except Exception as e:
            logger.info(f"Chunking KO - {e}")
            raise AssertionError(e) from e
        return ingestor.chunks_output, "Prêt à apprendre 🟢"
    except Exception as e:
        return e


def build_kernel() -> Kernel:
    """
    Builds a kernel by adding a chat service, an embedding service, and a memory plugin.

    Returns:
        Kernel: The kernel with the added services and plugin.

    Raises:
        AssertionError: If an error occurs during any of the service or plugin addition steps.
    """

    try:
        chat = AzureChatCompletion(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_ID"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        if os.getenv("AZURE_OPENAI_DEPLOYMENT_ID") in kernel.services:
            kernel.remove_service(os.getenv("AZURE_OPENAI_DEPLOYMENT_ID"))
            print("Replacing previous chat service")
            kernel.add_service(chat)
        else:
            kernel.add_service(chat)
    except Exception as e:
        raise AssertionError(f"FAIL - creating chat service - {e}") from e
    # Embedding
    try:
        embedding_service = AzureTextEmbedding(
            deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

        if os.getenv("AZURE_OPENAI_EMBEDDING") in kernel.services:
            # Remove the existing service
            kernel.remove_service(os.getenv("AZURE_OPENAI_EMBEDDING"))
            print("Replacing previous embedding service")
            kernel.add_service(embedding_service)
        else:
            kernel.add_service(embedding_service)
    except Exception as e:
        raise AssertionError(f"FAIL - embedding client - {e}") from e

    try:
        memory = SemanticTextMemory(storage=VolatileMemoryStore(), embeddings_generator=embedding_service)
        kernel.add_plugin(TextMemoryPlugin(memory), MEMORY_NAME)
    except Exception as e:
        raise AssertionError(f"FAIL - adding memory client - {e}") from e
    return kernel


def build_kernel_RAG(
    kernel: KernelFunction,
    AZURE_OPENAI_DEPLOYMENT_ID: str | None = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID"),
    VARS: list[InputVariable] = VARS,
    PROMPT: str = DEFAULT_RAG_PROMPT,
    FUNC_NAME: str = FUNC_NAME,
    PLUGIN_NAME: str = PLUGIN_NAME,
) -> KernelFunction | Exception:
    """
    This function builds a kernel with a RAG (Retriever-Augmented Generation) function attached.
    It creates a template configuration and attaches it to the kernel.

    Args:
        kernel (KernelFunction): The kernel to which the RAG function is to be attached.
        AZURE_OPENAI_DEPLOYMENT_ID (str, optional): The ID of the Azure OpenAI deployment. Defaults to AZURE_OPENAI_DEPLOYMENT_ID.
        VARS (dict, optional): Dictionary of variables for the prompt. Defaults to VARS.
        DEFAULT_RAG_PROMPT (str, optional): The default prompt for the RAG function. Defaults to DEFAULT_RAG_PROMPT.
        FUNC_NAME (str, optional): The name of the function to be attached to the kernel. Defaults to FUNC_NAME.
        PLUGIN_NAME (str, optional): The name of the plugin to be used. Defaults to PLUGIN_NAME.

    Returns:
        Union[KernelFunction, Exception]: The kernel with the RAG function attached if successful.
                                         An AssertionError with an attached message if an error occurred.
    """
    try:
        template_cfg = create_template_cfg(
            kernel, service=AZURE_OPENAI_DEPLOYMENT_ID, variables_prompt=VARS, prompt=PROMPT
        )
        kernel_rag = attach_cfg_func(kernel, FUNC_NAME, PLUGIN_NAME, template_cfg)
        return kernel_rag
    except Exception as e:
        raise AssertionError(f"FAIL - creating RAG function {e}") from e


def build_kernel_Classif(
    kernel: KernelFunction,
    AZURE_OPENAI_DEPLOYMENT_ID: str | None = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID"),
    VARS: list[InputVariable] = CLASSIF_VARS,
    PROMPT: str = DEFAULT_CLASSIFY_PROMPT,
    FUNC_NAME: str = CLASSIF_NAME,
    PLUGIN_NAME: str = CLASSIF_PLUGIN_NAME,
) -> KernelFunction | Exception:
    """
    This function builds a kernel with a Classify function attached.
    It creates a template configuration and attaches it to the kernel.

    Args:
        kernel (KernelFunction): The kernel to which the function is to be attached.
        AZURE_OPENAI_DEPLOYMENT_ID (str, optional): The ID of the Azure OpenAI deployment. Defaults to AZURE_OPENAI_DEPLOYMENT_ID.
        VARS (dict, optional): Dictionary of variables for the prompt. Defaults to VARS.
        PROMPT (str, optional): The default prompt for the Classif function.
        FUNC_NAME (str, optional): The name of the function to be attached to the kernel. Defaults to FUNC_NAME.
        PLUGIN_NAME (str, optional): The name of the plugin to be used. Defaults to PLUGIN_NAME.

    Returns:
        Union[KernelFunction, Exception]: The kernel with the function attached if successful.
                                         An AssertionError with an attached message if an error occurred.
    """
    try:
        template_cfg = create_template_cfg(
            kernel, service=AZURE_OPENAI_DEPLOYMENT_ID, variables_prompt=VARS, prompt=PROMPT
        )
        kernel_rag = attach_cfg_func(kernel, FUNC_NAME, PLUGIN_NAME, template_cfg)
        return kernel_rag
    except Exception as e:
        raise AssertionError(f"FAIL - creating Classif function {e}") from e


def new_kernel() -> Kernel | ValueError:
    """
    Reinitialize the kernel

    Args:
        None
    Returns:

    """
    try:
        new_kernel = build_kernel()
        build_kernel_RAG(new_kernel)
        build_kernel_Classif(new_kernel)
        return new_kernel
    except Exception as e:
        raise ValueError(f"Kernel inactif - {str(e)} 🔴") from e


async def create_memory(
    texts: str | dict,
    collection_name: str,
    skr: Kernel = kernel,
    memory_plugin: str = MEMORY_NAME,
    wait_param: int = 2,
    wait_step: int = 2,
) -> str:
    """
    Asynchronously creates a memory for a given kernel by adding chunks of text.

    If the input text is a string, it is converted to a dictionary before processing.
    The function then iterates through the items in the dictionary and adds each chunk to the memory.

    Args:
        texts (Union[str, dict]): The texts to be added to the memory. Can be a string or a dictionary.
        collection_name (str): The name of the collection.
        skr (Kernel, optional): The kernel to which the memory is added. Defaults to 'kernel'.
        memory_plugin (str, optional): The name of the memory plugin. Defaults to "TextMemoryPlugin".
        wait_step (int) : waiting time every "wait_step" documents in seconds, Default to 2
        wait_time (int) : wait every n documents, Default to 2

    Returns:
        str: Success message.

    Raises:
        AssertionError: If an error occurs during the string-to-dictionary conversion or while adding chunks to memory.
    """
    logger.info(f"Type input for memory {type(texts)} - content {texts}")
    try:
        if isinstance(texts, str):
            texts: dict = ast.literal_eval(texts)
            logger.info("Converts string to dict of chunks OK")
            logger.info(
                f"Docs = {texts.keys()} - with content {len([doc for doc,c in texts.items() if c['full_text']])} and {sum([len(c['full_text']) for c in texts.values() if c['full_text']])}"
            )
        elif isinstance(texts, dict):
            logger.info(
                f"Docs = {texts.keys()} - with content {len([doc for doc,c in texts.items() if c['full_text']])} and {sum([len(c['full_text']) for c in texts.values() if c['full_text']])}"
            )
        else:
            logger.info(f"Invalid input - {type(texts)}")
    except Exception as e:
        raise AssertionError(f"Converts string to dict of chunks KO - {e}") from e

    try:
        idd = 0
        logger.info(
            f"Docs = {texts.keys()} - with content {len([doc for doc,c in texts.items() if c['full_text']])} and {sum([len(c['full_text']) for c in texts.values() if c['full_text']])}"
        )
        for doc, content in texts.items():
            if content["full_text"]:
                for page, chunks in content["full_text"].items():
                    for idc, chunk in enumerate(chunks):
                        try:
                            await skr.get_function(memory_plugin, "save").invoke(
                                skr, text=chunk, key=f"{doc}-{page}-{idc}", collection=collection_name
                            )
                            logger.info("Adding chunks to memory OK")
                        except Exception as e:
                            raise AssertionError(f"Fail adding chunk to memory - {e}") from e
                if idd % wait_step == 0:
                    logger.info(f"Time sleep in creating memory - step {idd}")
                    await asyncio.sleep(wait_param)
    except Exception as e:
        raise AssertionError(f"Adding chunks to memory KO - {e}") from e

    return "Prêt à vous répondre 🟢"


async def send_query_RAG(
    query: str,
    collection_id: str,
    kernel: ker.Kernel = kernel,
    relevance_score: float = 0.8,
    name_plugin_memory: str = "TextMemoryPlugin",
    PLUGIN_NAME: str = PLUGIN_NAME,
    FUNC_NAME: str = FUNC_NAME,
    wait_param: int = 0,
) -> str:
    """
    Asynchronously sends a query to a Retriever-Augmented Generation (RAG) model and retrieves an answer.

    Args:
        query (str): The query to be sent.
        collection_id (str): The ID of the collection.
        kernel (ker.Kernel, optional): The kernel being used. Defaults to 'kernel'.
        relevance_score (float, optional): The relevance score threshold. Defaults to 0.8.
        name_plugin_memory (str, optional): The name of the memory plugin. Defaults to "TextMemoryPlugin".
        PLUGIN_NAME (str, optional): The name of the plugin. Defaults to PLUGIN_NAME.
        FUNC_NAME (str, optional): The name of the function. Defaults to FUNC_NAME.
        wait_param (int, optional): The amount of time to wait before fetching the answer. Defaults to 0.

    Returns:
        str: The response from the RAG model.

    Raises:
        Exception: If an error occurs while retrieving information from memory or generating the answer.
    """
    try:
        logger.info("Retrieve information from memory")
        context = await kernel.get_function(name_plugin_memory, "recall")(
            kernel=kernel, ask=query, collection=collection_id, relevance=relevance_score, limit=10
        )
    except Exception as e:
        logger.info(f"Retrieve information from memory KO  - {e}")
    try:
        logger.info("Generate Retrieve Augmented Answer OK")
        answer = await kernel.get_function(PLUGIN_NAME, FUNC_NAME)(kernel=kernel, question=query, context=context.value)
        await asyncio.sleep(wait_param)
        final_answer = answer.value[0]
        return final_answer
    except Exception as e:
        logger.info(f"Generate Retrieve Augmented Answer KO - {e}")


def classify_corpus(
    AO: gr.JSON, questions: gr.JSON | dict, define_task: gr.Textbox | str, modalites: gr.JSON | dict
) -> str:
    """
    Classifies a corpus of text based on a defined task and modalities.

    Args:
        AO (gr.JSON): A dictionary containing question-answer pairs.
        questions (gr.JSON | dict): A dictionary containing a set of questions to be filtered.
        define_task (gr.Textbox | str): A text input defining the task to be performed.
        modalites (gr.JSON | dict): A dictionary containing the modalities used for classification.

    Returns:
        str: A string response from the logical classifier, strictly answering with the provided modalities.

    This function filters the question-answer pairs based on the provided questions, defines a task and the modalities
    for classification, and returns the response of the logical classifier.
    """
    subset = [
        f"La question filtrée est {q} et la réponse correspondante est {a}"
        for q, a in AO.items()
        if q in questions["questions"]
    ]
    prompt = DEFAULT_CLASSIFY_PROMPT.format(define_task, " , ".join(modalites["modalites"]), subset)
    logger.info(prompt)
    res = send_prompt("Tu es un classifieur logique qui répond strictement par les modalités fournies", prompt)
    return res


async def fill_automatically(
    df_UI: pd.DataFrame | None, NomCorpus: gr.Textbox, df_API: gr.JSON | None, wait_param: int = 0, wait_step: int = 20
) -> list:
    """
    Asynchronously fills a dataframe with answers retrieved from a Retriever-Augmented Generation (RAG) model.

    The function takes questions either from a frontend DataFrame or a backend API, then sends each question to the RAG model.

    Args:
        df_UI (Union[pd.DataFrame, None]): The frontend DataFrame containing the questions.
        NomCorpus (gr.Textbox): The name of the corpus.
        df_API (Union[gr.JSON, None], optional): The backend API containing the questions. Defaults to None.
        wait_param (int, optional): The amount of time to wait before fetching the answer. Defaults to 0.
        wait_step (int, optional): The frequency of wait intervals. Defaults to 20.

    Returns:
        list: A list of dictionaries where each dictionary contains a question and its corresponding answer.

    Raises:
        Exception: If an error occurs while importing questions or creating kernels.
    """
    try:
        res = []
        if (df_API is not None) and len(df_API) > 0:
            channel = "backend"
            logger.info(f"Import questions OK from {channel}")
            df = pd.DataFrame(df_API, columns=["Question"])
        else:
            try:
                channel = "frontend"
                logger.info(f"Import questions OK from {channel}")
                df = df_UI
                logger.info(f"{df}")
            except Exception as e:
                raise ValueError("Please pass a valid question set") from e
    except Exception as e:
        logger.info(f"Import questions KO from {channel} - {e}")
    try:
        logger.info("Import questions OK from")
        tasks = []
        idc = 0
        for _, row in df.iterrows():
            idc += 1
            tasks.append(send_query_RAG(row["Question"], NomCorpus))
            if idc % wait_step == 0:
                logger.info("Time sleep in answering question")
                await asyncio.sleep(wait_param)
                idc = 0
        results = await asyncio.gather(*tasks)
        for question, result in zip(df["Question"], results, strict=False):
            res.append({question: result})
    except Exception as e:
        logger.info(f"Create kernels KO - {e}")
    return res


def update_questions() -> gr.Dropdown:
    """
    Met à jour et retourne un composant Dropdown de Gradio avec la liste des questions.

    Returns:
        gr.Dropdown: Un composant Dropdown interactif contenant la liste des questions.
    """
    return gr.Dropdown(ref_question.list_questions(), interactive=True)


def delete_questions(name: str) -> str | gr.Dropdown:
    """
    Supprime un référentiel de questions et met à jour le Dropdown.

    Args:
        name (str): Le nom du référentiel de questions à supprimer.

    Returns:
        str: Message confirmant la suppression du référentiel.
        gr.Dropdown: Un composant Dropdown mis à jour avec la liste des questions.
    """
    return ref_question.delete_questions(name), update_questions()


def save_questions(name: str, questions: gr.DataFrame) -> str | gr.Dropdown:
    """
    Sauvegarde un référentiel de questions et met à jour le Dropdown.

    Args:
        name (str): Le nom du référentiel de questions à sauvegarder.
        questions (gr.DataFrame): Un composant DataFrame de Gradio contenant les questions.

    Returns:
        str: Message confirmant la sauvegarde du référentiel.
        gr.Dropdown: Un composant Dropdown mis à jour avec la liste des questions.
    """
    return ref_question.save_questions(name, questions), update_questions()


def load_questions(name: str) -> gr.DataFrame:
    """
    Charge un référentiel de questions et retourne un composant DataFrame de Gradio.

    Args:
        name (str): Le nom du référentiel de questions à charger.

    Returns:
        gr.DataFrame: Un composant DataFrame de Gradio contenant les questions chargées.
    """
    logger.info(f"load_method : {ref_question.load_questions(name)}")
    return gr.DataFrame(ref_question.load_questions(name), headers=["Question"], label="❓")


if __name__ == "__main__":
    logger.info("Starting gradio UI")
    try:
        kernel = new_kernel()
        ref_question = Questions(PATH_SAVE_QUESTION)
    except Exception as e:
        logger.info(f"Create kernels KO - {e}")
    try:
        with gr.Blocks() as iface:
            with gr.Accordion("Documents 🗃️"):
                with gr.Row():
                    Corpus = gr.Files(
                        type="binary", file_count="multiple", file_types=[".pdf", ".xls", ".xlsx"], interactive=True
                    )
                    Corpus_API = gr.JSON(visible=False)
                    NomCorpus = gr.Text(placeholder="Nom du Corpus 🎯")
                with gr.Row():
                    CreateMemory = gr.Button("Charger les documents ✍🏼", interactive=True)
                    CreateRAG = gr.Button("Apprendre la leçon 👩🏼‍🎓", interactive=True)
                    ForgetRAG = gr.Button("Reprendre à 0 ♻️", interactive=True)
                StatusMemory = gr.Textbox("Données non chargées 🔴", show_label=False)
                StatusRAG = gr.Textbox("Leçon non apprise 🔴", show_label=False)
                StatusKernel = gr.Textbox("Kernel prêt l'emploi 🟢", show_label=False)
                ChunkedCorpus = gr.Text({}, visible=False)

            CreateMemory.click(fn=process_docs, inputs=[Corpus, Corpus_API], outputs=[ChunkedCorpus, StatusMemory])
            CreateRAG.click(create_memory, inputs=[ChunkedCorpus, NomCorpus], outputs=[StatusRAG])
            ForgetRAG.click(None, js="window.location.reload()")
            ForgetRAG.click(new_kernel, inputs=[], outputs=[])
            with gr.Accordion("Trame de questions"):
                Trame = gr.Dataframe(headers=["Question"], label="❓")
                Trame_API = gr.JSON(visible=False)
                with gr.Row():
                    with gr.Accordion("Gérer les questions ☄️", open=False):
                        with gr.Row():
                            NameQuestions = gr.Textbox(label="Nom du référentiel", interactive=True)
                            ExistingQuestions = gr.Dropdown(
                                ref_question.state, label="Choisir des questions", interactive=True
                            )
                            with gr.Column():
                                DeleteQuestions = gr.Button("Supprimer ces questions ❌")
                                LoadQuestions = gr.Button("Charger ces questions ☄️")
                                SaveQuestions = gr.Button("Sauvegarder ces questions 💾")

                AutoAsk = gr.Button("Lancer la détection automatique 🎯")
                Result_questions = gr.JSON()
                # Classification de la liasse
                Classify = gr.Button("Classer le Corpus", interactive=True, visible=False)
                Filtre_Classif = gr.JSON(visible=False)
                ClassifTask = gr.Textbox(visible=False)
                PossibleOutput = gr.JSON(visible=False)
                Result_Classif = gr.Textbox(visible=False)

            AutoAsk.click(fill_automatically, inputs=[Trame, NomCorpus, Trame_API], outputs=[Result_questions])
            Classify.click(
                classify_corpus,
                inputs=[Result_questions, Filtre_Classif, ClassifTask, PossibleOutput],
                outputs=[Result_Classif],
            )
            SaveQuestions.click(
                save_questions, inputs=[NameQuestions, Trame], outputs=[NameQuestions, ExistingQuestions]
            )

            LoadQuestions.click(load_questions, inputs=[ExistingQuestions], outputs=[Trame])
            DeleteQuestions.click(delete_questions, inputs=[NameQuestions], outputs=[NameQuestions, ExistingQuestions])

        iface.queue()
        iface.launch(
            server_name=GRADIO_SERVER_NAME,
            server_port=GRADIO_SERVER_PORT,
            show_error=True,  # pour le retour des logs dans les applications tiers qui utilisent gradio_client
        )
        logger.info(f"Closing Interface - server : {GRADIO_SERVER_NAME} - port {GRADIO_SERVER_PORT}")

    except Exception as e:
        logger.info(f"Create Interface KO - server : {GRADIO_SERVER_NAME} - port {GRADIO_SERVER_PORT} - {e}")
