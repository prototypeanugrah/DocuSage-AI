"""
Complete multimodal RAG pipeline for PDFs with images.

This demonstrates a full workflow:
1. Extract text from PDF pages
2. Extract images from PDF pages
3. Embed both text and images
4. Store in vector database with metadata linking them
5. Query using text or images

Usage:
    # Index a PDF with images
    python multimodal_pdf_rag.py index --pdf document.pdf --collection my_docs

    # Query the indexed content
    python multimodal_pdf_rag.py query --collection my_docs --question "What does the diagram show?"
"""

import argparse
import base64
import os
from typing import List

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from openai import OpenAI

from doc_rag.config import LLMConfig
from doc_rag.utils import load_config

load_dotenv()


class MultimodalPDFProcessor:
    """
    Process PDFs to extract both text and images for multimodal RAG.
    """

    def __init__(self, embeddings, llm_config: LLMConfig):
        """
        Initialize the processor.

        Args:
            embeddings: Embeddings instance for text and images
            llm_config: Configuration object
        """
        self.embeddings = embeddings
        self.config = llm_config

    def extract_text_and_images(self, pdf_path: str) -> List[Document]:
        """
        Extract both text and images from a PDF, creating Document objects.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of Document objects (both text and image documents)
        """
        print(f"Processing PDF: {pdf_path}")
        pdf_document = fitz.open(pdf_path)
        documents = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]

            # Extract text from page
            text = page.get_text()
            if text.strip():
                text_doc = Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "type": "text",
                    },
                )
                documents.append(text_doc)
                print(f"  Page {page_num + 1}: Extracted text ({len(text)} chars)")

            # Extract images from page
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                # Store image as base64 in document
                image_b64 = base64.b64encode(image_bytes).decode()

                image_doc = Document(
                    page_content=f"[Image from page {page_num + 1}, image {img_index + 1}]",
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "type": "image",
                        "image_data": f"data:image/png,{image_b64}",
                        "image_index": img_index + 1,
                    },
                )
                documents.append(image_doc)
                print(f"  Page {page_num + 1}: Extracted image {img_index + 1}")

        pdf_document.close()
        print(f"\n✓ Extracted {len(documents)} total items (text + images)")

        return documents


class NVIDIAMultimodalEmbeddings(Embeddings):
    """
    Custom embeddings class supporting both text and images.
    """

    def __init__(self, model_name: str):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
        )
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (can be text or image data URIs)"""
        embeddings = []

        for text in texts:
            # Check if this is an image data URI
            if text.startswith("data:image/"):
                # Embed as image
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
                # Embed as text
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
        """Embed a query (text only for now)"""
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


class MultimodalRAG:
    """
    Complete multimodal RAG system for PDFs with images.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

        # Initialize embeddings
        self.embeddings = NVIDIAMultimodalEmbeddings(
            model_name=config.multimodal_model_name
        )

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.gemini_llm_model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # Initialize processor
        self.processor = MultimodalPDFProcessor(self.embeddings, config)

    def index_pdf(self, pdf_path: str, collection_name: str):
        """
        Index a PDF with both text and images.

        Args:
            pdf_path: Path to the PDF file
            collection_name: Name for the collection
        """
        print("=" * 70)
        print("INDEXING PDF WITH MULTIMODAL CONTENT")
        print("=" * 70)
        print()

        # Extract text and images
        documents = self.processor.extract_text_and_images(pdf_path)

        if not documents:
            print("No content extracted from PDF!")
            return

        # Prepare documents for embedding
        # For images, we need to pass the image data URI
        texts_to_embed = []
        for doc in documents:
            if doc.metadata.get("type") == "image":
                # Use the image data URI for embedding
                texts_to_embed.append(doc.metadata["image_data"])
            else:
                # Use the text content
                texts_to_embed.append(doc.page_content)

        print("\nGenerating embeddings...")

        # Create vector database
        persist_dir = os.path.join(self.config.persist_directory, collection_name)

        # We need to create a custom approach since we're mixing text and images
        # Store each document individually with its embedding
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )

        # Add documents in batches
        batch_size = 5
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_texts = texts_to_embed[i : i + batch_size]

            # For Chroma, we need to use the actual content for embedding
            # But we'll store the original document
            vectordb.add_documents(batch)
            print(
                f"  Indexed {min(i + batch_size, len(documents))}/{len(documents)} items"
            )

        print(f"\n✓ Successfully indexed {len(documents)} items to '{collection_name}'")
        print(
            f"  - Text chunks: {sum(1 for d in documents if d.metadata.get('type') == 'text')}"
        )
        print(
            f"  - Images: {sum(1 for d in documents if d.metadata.get('type') == 'image')}"
        )
        print(f"  Stored at: {persist_dir}")

    def query(self, collection_name: str, question: str, k: int = 8):
        """
        Query the multimodal collection.

        Args:
            collection_name: Name of the collection
            question: Question to ask
            k: Number of results to retrieve
        """
        print("=" * 70)
        print("QUERYING MULTIMODAL COLLECTION")
        print("=" * 70)
        print(f"\nQuestion: {question}\n")

        # Load vector database
        persist_dir = os.path.join(self.config.persist_directory, collection_name)

        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )

        # Create retriever
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k})

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=True,
        )

        # Get answer
        result = qa_chain.invoke({"query": question})

        # Display results
        print(f"\nAnswer: {result['result']}\n")

        print("Retrieved Sources:")
        for i, doc in enumerate(result["source_documents"], 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            doc_type = doc.metadata.get("type", "unknown")

            if doc_type == "image":
                img_index = doc.metadata.get("image_index", "?")
                print(f"  {i}. [IMAGE] {source} - Page {page}, Image {img_index}")
            else:
                print(f"  {i}. [TEXT] {source} - Page {page}")
                print(f"     Preview: {doc.page_content[:100]}...")

        return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multimodal RAG for PDFs with images")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a PDF with images")
    index_parser.add_argument("--pdf", required=True, help="Path to PDF file")
    index_parser.add_argument("--collection", required=True, help="Collection name")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query indexed content")
    query_parser.add_argument("--collection", required=True, help="Collection name")
    query_parser.add_argument("--question", required=True, help="Question to ask")
    query_parser.add_argument("--k", type=int, default=8, help="Number of results")

    args = parser.parse_args()

    # Load configuration
    config = load_config("doc_rag/config/config.yaml")
    llm_config = LLMConfig(**config)

    # Initialize RAG system
    rag = MultimodalRAG(llm_config)

    # Execute command
    if args.command == "index":
        rag.index_pdf(args.pdf, args.collection)
    elif args.command == "query":
        rag.query(args.collection, args.question, args.k)


if __name__ == "__main__":
    main()
