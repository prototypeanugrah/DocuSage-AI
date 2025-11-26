from typing import List, Optional

import arxiv
from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    text: str
    metadata: Optional[dict] = None


class DocumentList(BaseModel):
    documents: List[Document]


class DataIngestionConfig(BaseModel):
    query: str = Field(default="cs.CV", description="The query to search for")
    max_results: int = Field(
        default=5, description="The maximum number of results to return"
    )
    sort_by: arxiv.SortCriterion = Field(
        default=arxiv.SortCriterion.SubmittedDate,
        description="The field to sort the results by",
    )
    sort_order: arxiv.SortOrder = Field(
        default=arxiv.SortOrder.Descending,
        description="The order to sort the results in",
    )
    discovered_papers_save_dir: str = Field(
        default="src/arxiv_rag/artifacts/papers",
        description="The directory to save the discovered papers to",
    )


class LLMConfig(BaseModel):
    chunk_size: int = Field(
        default=500, description="The size of the chunks to split the text into"
    )
    chunk_overlap: int = Field(
        default=200, description="The overlap between the chunks"
    )
    max_tokens: int = Field(
        default=100, description="The maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.0, description="The temperature to use for the LLM"
    )
    persist_directory: str = Field(
        default="db", description="The directory to save the vector database to"
    )
    model: str = Field(default="gemini", description="The model to use for the LLM")
    max_retries: int = Field(
        default=3, description="The maximum number of retries to make"
    )
    llm_model_name: str = Field(
        default="gpt-4o-mini", description="The model to use for the LLM"
    )
    multimodal_embedding_model_name: str = Field(
        default="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
        description="The model to use for the multimodal embedding",
    )
    verbose: bool = Field(default=False, description="Whether to print verbose output")


class Config(BaseModel):
    data_ingestion: DataIngestionConfig
    llm_config: LLMConfig
