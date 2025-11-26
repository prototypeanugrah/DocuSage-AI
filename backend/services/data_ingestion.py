import os
from pathlib import Path

import arxiv
from rich.console import Console

from backend.config.schema import DataIngestionConfig
from backend.utils.utils import load_config

console = Console()


def arxiv_query(query: str, max_results: int) -> list[arxiv.Result]:
    """Query the Arxiv API for papers.

    Args:
        query (str): The query to search for.
        max_results (int): The maximum number of results to return.

    Returns:
        list[arxiv.Result]: The list of results.
    """
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results)
    return client.results(search)


def data_ingestion(config: DataIngestionConfig) -> str:
    # Load the config
    # Query the Arxiv API for papers
    search = arxiv_query(
        config.query,
        config.max_results,
    )

    # List of missing papers
    missing_papers = []

    # Iterate over the results
    for idx, result in enumerate(search):
        try:
            console.print(
                f"Processing paper {idx + 1}: {result.title} ({result.entry_id})",
                style="blue",
            )
            paper_name = f"{result.entry_id.split('/')[-1]}"

            # console.print("%(result)s", {"result": result}, style="white")
            # console.print("Title: %(title)s", {"title": result.title}, style="white")
            # console.print(
            #     "Summary: %(summary)s", {"summary": result.summary}, style="white"
            # )

            # if the file is not available, download the pdf in the src/artifacts directory
            os.makedirs(config.discovered_papers_save_dir, exist_ok=True)
            if not os.path.exists(
                os.path.join(config.discovered_papers_save_dir, paper_name + ".pdf")
            ):
                result.download_pdf(
                    dirpath=config.discovered_papers_save_dir,
                    filename=paper_name + ".pdf",
                )
                console.print(
                    "Downloaded paper %(paper_name)s to %(save_dir)s",
                    {
                        "paper_name": paper_name,
                        "save_dir": config.discovered_papers_save_dir,
                    },
                    style="green",
                )
            else:
                console.print(
                    f"Paper {paper_name} already exists in {config.discovered_papers_save_dir}. Skipping download...",
                    style="yellow",
                )
        except Exception:
            missing_papers.append(result.pdf_url)

    return config.discovered_papers_save_dir


if __name__ == "__main__":
    console.print("Starting data ingestion...")

    config_path = Path("config/config.yaml")
    config = load_config(config_path)
    data_config = config.data_ingestion

    data_ingestion(config=data_config)
    console.print("Data ingestion completed successfully.")
