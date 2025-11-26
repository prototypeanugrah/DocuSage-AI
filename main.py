"""
Smart PDF RAG with automatic image detection and routing.

This solution:
1. Checks each PDF page for images
2. Routes pages WITH images → multimodal embedding (text + image)
3. Routes pages WITHOUT images → text-only embedding
4. Uses the same NVIDIA model for both paths

Usage:
    # Index a PDF (automatically detects and routes pages)
    python smart_pdf_rag.py index --pdf document.pdf --collection my_docs

    # Query the indexed content
    python smart_pdf_rag.py query --collection my_docs --question "Your question here"
"""

import argparse

from dotenv import load_dotenv
from rich.console import Console

from backend.config import Config, DataIngestionConfig, LLMConfig
from backend.services.data_ingestion import data_ingestion
from backend.services.rag_service import RAGService
from backend.utils import load_config

load_dotenv()

console = Console()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Smart PDF RAG with automatic image detection and routing"
    )

    parser.add_argument(
        "--config",
        required=False,
        default="backend/config/config.yaml",
        help="Path to config file",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    index_parser = subparsers.add_parser(
        "index",
        help="Index a PDF or directory of PDFs (automatically routes pages based on image presence)",
    )
    index_parser.add_argument(
        "--data", required=True, help="Path to PDF file or directory containing PDFs"
    )
    index_parser.add_argument("--collection", required=True, help="Collection name")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query indexed content")
    query_parser.add_argument("--collection", required=True, help="Collection name")
    query_parser.add_argument("--question", required=True, help="Question to ask")
    query_parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Number of results to retrieve (default: 6)",
    )

    args = parser.parse_args()

    # Load configuration
    config: Config = load_config(args.config)
    data_ingestion_config: DataIngestionConfig = config.data_ingestion
    llm_config: LLMConfig = config.llm_config

    # Initialize smart RAG system
    rag = RAGService(llm_config)

    # # Execute command
    if args.command == "ingestion":
        console.print(
            f"{'=' * 20} Starting data ingestion... {'=' * 20}", style="green"
        )
        data_ingestion(data_ingestion_config)
        console.print(
            f"{'=' * 20} Data ingestion completed successfully. {'=' * 20}",
            style="green",
        )
    elif args.command == "index":
        console.print(f"{'=' * 20} Starting indexing... {'=' * 20}", style="green")
        rag.index(args.data, args.collection)
        console.print(
            f"{'=' * 20} Indexing completed successfully. {'=' * 20}", style="green"
        )
    elif args.command == "query":
        console.print(f"{'=' * 20} Starting querying... {'=' * 20}", style="green")
        rag.query(args.collection, args.question, args.k)
        console.print(
            f"{'=' * 20} Querying completed successfully. {'=' * 20}", style="green"
        )


if __name__ == "__main__":
    main()
