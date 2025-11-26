import os
import re
from time import time
from typing import Iterable, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from rich.console import Console
from tqdm import tqdm

from backend.config import LLMConfig
from backend.services.embedding_service import NVIDIAEmbeddings
from backend.services.pdf_processor import PDFProcessor

console = Console()


class RAGService:
    """
    RAG service with automatic routing.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        # Normalize persist directory to an absolute path so CWD doesn't change behavior
        self.persist_directory = os.path.abspath(config.persist_directory)

        # Initialize embeddings (same model, automatic routing)
        self.embeddings = NVIDIAEmbeddings(
            model_name=config.multimodal_embedding_model_name
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.llm_model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=20,
        )

        self.verbose = config.verbose

        # Initialize processor
        self.processor = PDFProcessor(config, verbose=self.verbose)

    @staticmethod
    def _extract_keywords(question: str) -> set[str]:
        """Lightweight keyword extractor to spot author or title tokens."""
        stop_words = {
            "paper",
            "summarize",
            "summarise",
            "summary",
            "about",
            "question",
            "tell",
            "explain",
            "describe",
            "give",
            "by",
            "the",
            "for",
            "of",
            "a",
            "an",
        }
        tokens = re.findall(r"[a-zA-Z][a-zA-Z]+", question.lower())
        return {t for t in tokens if len(t) > 3 and t not in stop_words}

    @staticmethod
    def _doc_text(doc: Document) -> str:
        """Return text that should be used for lexical matching."""
        if doc.metadata.get("type") == "multimodal":
            return (doc.metadata.get("text_context") or "").lower()
        return (doc.page_content or "").lower()

    def _choose_focus_source(
        self,
        docs_with_scores: Sequence[Tuple[Document, float]],
        question: str,
        persist_dir: str,
        collection_name: str,
    ) -> Tuple[str | None, int]:
        """Pick the best source based on keyword overlap. Falls back to a full scan if needed."""
        keywords = self._extract_keywords(question)
        best_source = None
        best_hits = 0

        for doc, _ in docs_with_scores:
            text = self._doc_text(doc)
            if not text:
                continue

            hits = sum(1 for kw in keywords if kw in text)
            if hits > best_hits:
                best_hits = hits
                best_source = doc.metadata.get("source")

        if best_hits == 0 and keywords:
            # Fallback: lightweight lexical scan over the whole collection
            client = QdrantClient(path=persist_dir)
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=collection_name,
                    limit=256,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
                for point in points:
                    payload = point.payload or {}
                    meta = payload.get("metadata") or {}
                    text = payload.get("page_content") or ""
                    if meta.get("type") == "multimodal":
                        text = meta.get("text_context") or ""
                    text = text.lower()
                    hits = sum(1 for kw in keywords if kw in text)
                    if hits > best_hits:
                        best_hits = hits
                        best_source = meta.get("source")

                if offset is None:
                    break

        if not best_source and docs_with_scores:
            best_source = docs_with_scores[0][0].metadata.get("source")

        return best_source, best_hits

    @staticmethod
    def _format_documents(docs: Iterable[Document]) -> str:
        """Flatten documents into a readable context block."""
        formatted = []
        for doc in docs:
            meta = doc.metadata or {}
            source = meta.get("source", "Unknown")
            page = meta.get("page", "?")
            doc_type = meta.get("type", "text")

            if doc_type == "multimodal":
                content = meta.get("text_context") or "[Image-only content]"
            else:
                content = doc.page_content

            formatted.append(f"Source: {source} | Page: {page}\n{content}")

        return "\n\n".join(formatted)

    def index(self, data_path: str, collection_name: str):
        """
        Index a PDF or directory of PDFs with automatic routing.

        Args:
            data_path: Path to PDF file or directory containing PDFs
            collection_name: Name for the collection
        """

        start_time = time()

        # Determine files to process
        pdf_files = []
        if os.path.isdir(data_path):
            # Get all PDF files in directory
            for root, _, files in os.walk(data_path):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        pdf_files.append(os.path.join(root, file))
            # console.print(
            #     f"Found {len(pdf_files)} PDF files in directory: {data_path}",
            #     style="blue",
            # )
        else:
            # Single file
            if data_path.lower().endswith(".pdf"):
                pdf_files.append(data_path)
            else:
                console.print(f"❌ Error: {data_path} is not a PDF file", style="red")
                return

        if not pdf_files:
            console.print("❌ No PDF files found to index!", style="red")
            return

        all_documents = []
        total_stats = {
            "total_pages": 0,
            "pages_with_images": 0,
            "pages_text_only": 0,
            "total_images": 0,
            "text_chunks": 0,
        }

        # Process each PDF
        for _, pdf_file in tqdm(
            enumerate(pdf_files, 1),
            total=len(pdf_files),
            desc="Processing PDFs",
            colour="blue",
        ):
            documents, stats = self.processor.process_pdf(pdf_file)

            if documents:
                all_documents.extend(documents)
                # Aggregate stats
                for key in total_stats:
                    total_stats[key] += stats.get(key, 0)

        if not all_documents:
            console.print("❌ No content extracted from any files!", style="red")
            return

        # Create vector database
        persist_dir = os.path.join(self.persist_directory, collection_name)

        # Add documents to Qdrant
        console.print(
            "Generating embeddings and indexing documents... (this may take a while)",
            style="yellow",
        )
        vectordb = QdrantVectorStore.from_documents(
            all_documents,
            self.embeddings,
            path=persist_dir,
            collection_name=collection_name,
        )

        console.print(
            f"✅ Successfully indexed {len(pdf_files)} files to collection '{collection_name}'",
            style="green",
        )
        console.print(
            f"📊 Total stats: {total_stats['total_images']} images, {total_stats['text_chunks']} text chunks",
            style="blue",
        )

        end_time = time()
        console.print(f"Time taken: {end_time - start_time} seconds", style="green")

    def query(self, collection_name: str, question: str, k: int = 8):
        """
        Query the indexed collection.

        Args:
            collection_name: Name of the collection
            question: Question to ask
            k: Number of results to retrieve
        """
        console.print(f"\n{'-' * 20}", style="blue")
        console.print("QUERYING COLLECTION", style="blue")
        console.print(f"{'-' * 20}", style="blue")
        console.print(f"Collection: {collection_name}", style="blue")
        console.print(f"Question: {question}\n", style="blue")

        persist_dir = os.path.join(self.persist_directory, collection_name)

        # Basic validation: ensure collection path exists
        if not os.path.exists(persist_dir):
            raise ValueError(f"Collection '{collection_name}' not found at {persist_dir}")

        # Load vector database
        try:
            vectordb = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                collection_name=collection_name,
                path=persist_dir,
            )
        except Exception as exc:
            raise ValueError(f"Failed to load collection '{collection_name}': {exc}") from exc

        search_k = max(k, 6)

        found_docs_with_scores = vectordb.similarity_search_with_score(
            question, k=search_k
        )

        if not found_docs_with_scores:
            console.print("❌ No results returned from the vector store.", style="red")
            raise ValueError("No results returned from the vector store")

        focus_source, keyword_hits = self._choose_focus_source(
            found_docs_with_scores, question, persist_dir, collection_name
        )

        filtered_docs: list[Document] = []
        if focus_source:
            source_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.source",
                        match=qdrant_models.MatchValue(value=focus_source),
                    )
                ]
            )
            filtered_docs = vectordb.similarity_search(
                question, k=max(search_k, k * 2), filter=source_filter
            )

        # Fall back to the highest scoring docs if source-filtered set is empty
        retrieved_docs: list[Document] = filtered_docs or [
            doc for doc, _ in found_docs_with_scores[:search_k]
        ]

        context_block = self._format_documents(retrieved_docs)

        # Prompt
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        # Create QA chain
        qa_chain = prompt | self.llm
        # Get answer
        result = qa_chain.invoke({"context": context_block, "question": question})

        # Display results
        console.print(f"\n{'-' * 20}", style="blue")
        console.print("ANSWER", style="blue")
        console.print(f"{'-' * 20}", style="blue")
        console.print(f"{result.content}\n", style="blue")

        console.print(f"{'-' * 20}", style="blue")
        console.print("SOURCES", style="blue")
        console.print(f"{'-' * 20}", style="blue")
        if focus_source and keyword_hits:
            console.print(
                f"Focused on source '{focus_source}' based on keyword match.",
                style="blue",
            )
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            doc_type = doc.metadata.get("type", "unknown")
            has_image = doc.metadata.get("has_image", False)

            if doc_type == "multimodal":
                img_index = doc.metadata.get("image_index", "?")
                console.print(
                    f"{i}. 🖼️  [IMAGE] {source} - Page {page}, Image {img_index}",
                    style="blue",
                )
                text_context = doc.metadata.get("text_context", "")
                if text_context:
                    console.print(f"   Context: {text_context[:100]}...", style="blue")
            else:
                icon = "📄" if not has_image else "📝"
                console.print(
                    f"{i}. {icon} [TEXT] {source} - Page {page}", style="blue"
                )
                console.print(f"   {doc.page_content[:120]}...", style="blue")
            console.print()

        return {
            "answer": result.content,
            "sources": retrieved_docs,
            "focus_source": focus_source if keyword_hits else None
        }
