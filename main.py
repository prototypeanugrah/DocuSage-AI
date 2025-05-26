"""
This is a RAGTool for indexing and querying documents.
It uses the Google Generative AI API to index and query documents.
It uses the Chroma vector database to store and query documents.
It uses the Langchain framework to build the RAG pipeline.

Usage:
python main.py index
    --config config.yaml
    <path to file or directory to index>
    --metadata <name of the collection>
python main.py ask
    --config config.yaml
    <name of the collection>
    <question>
"""

import argparse
import logging
import os
import time
from typing import Any, Dict, List, Optional

import dotenv
import langchain_core
import yaml  # Add this import after other imports
from google import genai
from google.genai import types
from langchain.chains import RetrievalQA

# Anthropic imports
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma

# import typer  # Removed Typer
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document

# Text splitters imports
from langchain_experimental.text_splitter import SemanticChunker

# Gemini imports
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

# Ollama imports
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

# OpenAI imports
from langchain_openai import OpenAI, OpenAIEmbeddings
from pydantic import BaseModel
from rich import print

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


class RAGConfig(BaseModel):
    """
    RAGConfig is a configuration for the RAGTool.
    It contains the configuration keys and their datatypes and default values.

    Args:
        BaseModel: The base model for the RAGConfig.
    """

    chunk_size: int
    chunk_overlap: int
    max_tokens: int
    temperature: float
    persist_directory: str
    model: str
    max_retries: int
    openai_embedding_model_name: str
    ollama_embedding_model_name: str
    gemini_embedding_model_name: str
    openai_llm_model_name: str
    anthropic_llm_model_name: str
    ollama_llm_model_name: str
    gemini_llm_model_name: str


def model_factory(
    model_name: str,
    config: RAGConfig,
) -> tuple[Any, Any, Any]:
    """
    This function is used to create the embeddings and llm models depending
    on the model_name.

    Args:
        model_name (str): The name of the model to use.
        config (RAGConfig): The configuration for the RAGTool.

    Raises:
        ValueError: If the model_name is not valid.

    Returns:
        tuple: A tuple containing the embeddings, llm, and client.
    """
    if model_name == "gemini":
        embeddings = GoogleGenerativeAIEmbeddings(
            model=config.gemini_embedding_model_name
        )
        llm = ChatGoogleGenerativeAI(
            api_key=GOOGLE_API_KEY,
            model=config.gemini_llm_model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        client = genai.Client()
    elif model_name == "openai":
        embeddings = OpenAIEmbeddings(model=config.openai_embedding_model_name)
        llm = OpenAI(
            api_key=OPENAI_API_KEY,
            model=config.openai_llm_model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_retries=config.max_retries,
        )
        client = None
    elif model_name == "ollama":
        embeddings = OllamaEmbeddings(model=config.ollama_embedding_model_name)
        llm = ChatOllama(
            model=config.ollama_llm_model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        client = None
    elif model_name == "claude":
        embeddings = OllamaEmbeddings(model=config.ollama_embedding_model_name)
        llm = ChatAnthropic(
            api_key=ANTHROPIC_API_KEY,
            model=config.anthropic_llm_model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_retries=config.max_retries,
        )
        client = None
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return embeddings, llm, client


class RAGTool:
    """
    Tool for indexing and querying documents using a RAG pipeline.
    """

    def __init__(self, config: RAGConfig) -> None:
        """
        Initialize the RAGTool.

        Args:
            config (RAGConfig): The configuration for the RAGTool.
        """
        self.config = config
        self.embeddings, self.llm, self.client = model_factory(
            config.model.lower(), config
        )

    def get_loaders(self, path: str) -> List[BaseLoader]:
        """
        Get the loaders for the RAGTool.

        Args:
            path (str): The path to the documents to load.

        Returns:
            list[langchain_core.document_loaders.base.BaseLoader]: The loaders
            for the RAGTool.
        """
        loaders: List[BaseLoader] = []
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.startswith("~$"):
                        continue  # Skip temp/lock files
                    full_path = os.path.join(root, file)
                    ext = os.path.splitext(full_path)[1].lower()
                    try:
                        if ext == ".pdf":
                            loaders.append(PyPDFLoader(full_path))
                        elif ext in [".docx", ".doc"]:
                            loaders.append(UnstructuredWordDocumentLoader(full_path))
                        elif ext in [".pptx", ".ppt"]:
                            loaders.append(UnstructuredPowerPointLoader(full_path))
                        elif ext == ".txt":
                            loaders.append(TextLoader(full_path, encoding="utf8"))
                    except ValueError as e:
                        logger.error("Skipping file %s: %s", file, e)
        else:
            ext = os.path.splitext(path)[1].lower()
            try:
                if ext == ".pdf":
                    loaders.append(PyPDFLoader(path))
                elif ext in [".docx", ".doc"]:
                    loaders.append(UnstructuredWordDocumentLoader(path))
                elif ext in [".pptx", ".ppt"]:
                    loaders.append(UnstructuredPowerPointLoader(path))
                elif ext == ".txt":
                    loaders.append(TextLoader(path, encoding="utf8"))
                else:
                    raise ValueError(f"Unsupported file type: {ext}")
            except ValueError as e:
                logger.error("Skipping file %s: %s", os.path.basename(path), e)
        return loaders

    def _load_and_tag_documents(self, path: str) -> List[Document]:
        """
        Index the documents in the RAGTool.

        Args:
            path (str): The path to the documents to index.
            metadata (Optional[str]): The metadata for the documents.

        Returns:
            None
        """
        all_docs: List[Document] = []
        for loader in self.get_loaders(path):
            docs = loader.load()
            src_path = getattr(loader, "file_path", path)
            for doc in docs:
                doc.metadata["source"] = os.path.basename(src_path)
            all_docs.extend(docs)
        return all_docs

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        """
        Split documents into chunks using a semantic chunker.
        """
        splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="standard_deviation",
            buffer_size=3,
        )
        return splitter.split_documents(docs)

    def index(self, path: str, metadata: Optional[str] = None) -> None:
        """
        Index the documents at the given path, optionally using a metadata key.
        """
        all_docs = self._load_and_tag_documents(path)
        split_docs = self._split_documents(all_docs)
        print(f"Total split documents: {len(split_docs)}")
        store_path = os.path.join(
            self.config.persist_directory, metadata or os.path.basename(path)
        )
        # Uncomment to persist to Chroma vector DB:
        vectordb = Chroma.from_documents(
            split_docs,
            self.embeddings,
            persist_directory=store_path,
            collection_name=metadata or os.path.basename(path),
        )
        logger.info("Indexed %s documents to %s", len(split_docs), store_path)

    def upload_and_cache_file(self, file_path: str) -> str:
        """
        Upload and cache a file in the RAGTool.

        Args:
            file_path (str): The path to the file to upload and cache.

        Returns:
            str: The name of the cached content.
        """
        # Upload file
        file = self.client.files.upload(file=file_path)
        while file.state.name == "PROCESSING":
            time.sleep(2)
            file = self.client.files.get(name=file.name)
        cache = self.client.caches.create(
            model=self.config.gemini_llm_model_name,
            config=types.CreateCachedContentConfig(
                display_name="Cached Content",
                system_instruction=(
                    "You are an expert content analyzer, and your job is to answer "
                    "the user's query based on the files you have access to."
                ),
                contents=[file],
                ttl="300s",
            ),
        )
        return cache.name

    def query_cached_content(self, cache_name: str, question: str) -> str:
        """
        Query the cached content in the RAGTool.

        Args:
            cache_name (str): The name of the cached content.
            question (str): The question to ask the cached content.

        Returns:
            str: The answer to the question.
        """
        llm = self.llm
        llm.cached_content = cache_name  # type: ignore[attr-defined]
        message = langchain_core.messages.HumanMessage(content=question)
        response = llm.invoke([message])
        return response

    def query(self, metadata: str, question: str) -> Dict[str, Any]:
        """
        Query the RAGTool.

        Args:
            metadata (str): The metadata for the documents.
            question (str): The question to ask the RAGTool.

        Returns:
            dict: The result containing the answer and source documents.
        """
        store_path = os.path.join(self.config.persist_directory, metadata)
        vectordb = Chroma(
            persist_directory=store_path,
            embedding_function=self.embeddings,
            collection_name=metadata,
        )
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8})

        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=True,
        )
        result = qa_chain.invoke({"query": question})

        answer = self.get_model_response(
            question, qa_chain
        )  # Get the answer from the model
        print(f"[bold green]Answer:[/bold green] {answer}")
        sources = result.get(
            "source_documents", []
        )  # Get all the sources from the result
        relevant_sources = self.get_model_sources(
            sources
        )  # Get the relevant sources from the model
        for source in relevant_sources:
            print(f"[bold green]Source:[/bold green] {source}")

        return {"answer": answer, "sources": relevant_sources}

    def get_model_response(self, question: str, rqa: RetrievalQA) -> str:
        """
        Get the response from the model.

        Args:
            question (str): The question to ask the model.
            rqa (RetrievalQA): The retrieval QA chain.

        Returns:
            str: The response from the model.
        """
        return rqa.invoke({"query": question}).get("result", "")

    def get_model_sources(self, sources: List[Document]) -> List[str]:
        """
        Get the sources from the model.

        Args:
            sources (List[Document]): The sources to get.

        Returns:
            List[str]: The sources from the model.
        """
        return set(
            doc.metadata.get("source") for doc in sources if doc.metadata.get("source")
        )


def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Tool to index and query documents"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    index_parser = subparsers.add_parser(
        "index", help="Index a PDF, doc, or folder for RAG."
    )
    index_parser.add_argument(
        "path", type=str, help="Path to file or directory to index"
    )
    index_parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Metadata name for the collection",
    )
    index_parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["openai", "gemini", "claude", "ollama"],
        help="Model to use: openai, gemini, claude, ollama",
    )

    # Ask command
    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question against an indexed RAG collection."
    )
    ask_parser.add_argument(
        "metadata", type=str, help="Metadata name for the collection"
    )
    ask_parser.add_argument("question", type=str, help="Question to ask")

    args = parser.parse_args()

    config_dict = load_config_from_yaml(args.config)

    config = RAGConfig(**config_dict)
    tool = RAGTool(config)

    if args.command == "index":
        tool.index(args.path, args.metadata)
    elif args.command == "ask":
        tool.query(args.metadata, args.question)


if __name__ == "__main__":
    main()
