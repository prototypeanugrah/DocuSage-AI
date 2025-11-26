import base64
import os
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

from backend.config import LLMConfig

console = Console()


class PDFProcessor:
    """
    Processes PDF pages intelligently:
    - Pages with images → multimodal embedding
    - Pages without images → text-only embedding
    """

    def __init__(self, config: LLMConfig, verbose: bool = False):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.verbose = verbose

    def process_pdf(self, pdf_path: str) -> Tuple[List[Document], Dict[str, int]]:
        """
        Process PDF and route pages based on image presence.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (documents, statistics)
        """
        if self.verbose:
            console.print(f"\n{'=' * 20}", style="blue")
            console.print(f"Processing PDF: {pdf_path}", style="blue")
            console.print(f"{'-' * 20}\n", style="blue")

        pdf_document = fitz.open(pdf_path)
        documents = []
        stats = {
            "total_pages": len(pdf_document),
            "pages_with_images": 0,
            "pages_text_only": 0,
            "total_images": 0,
            "text_chunks": 0,
        }

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_number = page_num + 1

            # Check if page has images
            image_list = page.get_images(full=True)
            has_images = len(image_list) > 0

            # Extract text
            text = page.get_text().strip()

            if has_images:
                # ROUTE: Multimodal path (text + images)
                stats["pages_with_images"] += 1
                stats["total_images"] += len(image_list)

                if self.verbose:
                    console.print(
                        f"📄 Page {page_number}: HAS IMAGES ({len(image_list)}) → Multimodal path",
                        style="purple",
                    )

                # Extract each image and create multimodal document
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_b64 = base64.b64encode(image_bytes).decode()

                    image_ext = base_image["ext"]
                    image_mime = (
                        f"image/{image_ext}" if image_ext != "jpg" else "image/jpeg"
                    )

                    # Create document with image data URI
                    # The embeddings class will detect this and use image embedding
                    doc = Document(
                        page_content=f"data:{image_mime};base64,{image_b64}",
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_number,
                            "type": "multimodal",
                            "has_image": True,
                            "image_index": img_index + 1,
                            "text_context": text[:500]
                            if text
                            else "",  # Store text context
                        },
                    )
                    documents.append(doc)
                    if self.verbose:
                        console.print(
                            f"   ✓ Image {img_index + 1} embedded with context",
                            style="purple",
                        )

                # Also add text chunks if there's substantial text
                if text and len(text) > 100:
                    text_chunks = self.text_splitter.split_text(text)
                    for chunk_idx, chunk in enumerate(text_chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": os.path.basename(pdf_path),
                                "page": page_number,
                                "type": "text",
                                "has_image": True,
                                "chunk": chunk_idx + 1,
                            },
                        )
                        documents.append(doc)
                    stats["text_chunks"] += len(text_chunks)

                    if self.verbose:
                        console.print(
                            f"   ✓ {len(text_chunks)} text chunks added",
                            style="purple",
                        )

            else:
                # ROUTE: Text-only path
                if text:
                    stats["pages_text_only"] += 1
                    if self.verbose:
                        console.print(
                            f"📄 Page {page_number}: TEXT ONLY → Text embedding path",
                            style="purple",
                        )

                    # Split text into chunks
                    text_chunks = self.text_splitter.split_text(text)

                    for chunk_idx, chunk in enumerate(text_chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": os.path.basename(pdf_path),
                                "page": page_number,
                                "type": "text",
                                "has_image": False,
                                "chunk": chunk_idx + 1,
                            },
                        )
                        documents.append(doc)

                    stats["text_chunks"] += len(text_chunks)
                    if self.verbose:
                        console.print(
                            f"   ✓ {len(text_chunks)} text chunks created",
                            style="purple",
                        )
                else:
                    if self.verbose:
                        console.print(
                            f"📄 Page {page_number}: EMPTY → Skipped",
                            style="purple",
                        )

        pdf_document.close()

        if self.verbose:
            console.print(f"\n{'-' * 20}", style="blue")
            console.print("PROCESSING SUMMARY", style="blue")
            console.print(f"{'-' * 20}", style="blue")
            console.print(f"Total pages: {stats['total_pages']}", style="blue")
            console.print(
                f"Pages with images: {stats['pages_with_images']} (multimodal path)",
                style="blue",
            )
            console.print(
                f"Pages text-only: {stats['pages_text_only']} (text path)", style="blue"
            )
            console.print(
                f"Total images embedded: {stats['total_images']}", style="blue"
            )
            console.print(f"Total text chunks: {stats['text_chunks']}", style="blue")
            console.print(f"Total documents created: {len(documents)}", style="blue")
            console.print(f"{'-' * 20}\n", style="blue")

        return documents, stats
