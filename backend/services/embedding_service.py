import os
from typing import List

from langchain_core.embeddings import Embeddings
from openai import OpenAI


class NVIDIAEmbeddings(Embeddings):
    """
    Smart embeddings class that automatically routes to text or image embedding
    based on content type, using the same NVIDIA multimodal model.
    """

    def __init__(self, model_name: str):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
        )
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents - automatically detects if content is text or image.

        Args:
            texts: List of text strings or image data URIs

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for text in texts:
            # Check if this is an image data URI
            if text.startswith("data:image/"):
                # Route to IMAGE embedding path
                response = self.client.embeddings.create(
                    input=[text],
                    model=self.model_name,
                    encoding_format="float",
                    extra_body={
                        "modality": ["image"],
                        "input_type": "passage",
                        "truncate": "NONE",
                    },
                )
            else:
                # Route to TEXT embedding path
                response = self.client.embeddings.create(
                    input=[text],
                    model=self.model_name,
                    encoding_format="float",
                    extra_body={
                        "modality": ["text"],
                        "input_type": "passage",
                        "truncate": "NONE",
                    },
                )

            embeddings.append(response.data[0].embedding)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query (text-only)"""
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name,
            encoding_format="float",
            extra_body={
                "modality": ["text"],
                "input_type": "query",
                "truncate": "NONE",
            },
        )
        return response.data[0].embedding
