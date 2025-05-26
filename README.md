# IntelliPDF Agent

This repository contains a project that leverages modern AI techniques to automate and enhance workflows for processing PDF documents. It is designed to extract insights, summarize content, and assist in various document-driven tasks.

## What is the IntelliPDF Agent?

The IntelliPDF Agent is a tool that helps users interact with PDF documents using artificial intelligence. It automates tasks such as extracting text, summarizing content, answering questions, and streamlining document-related workflows. Whether you're dealing with research papers, financial reports, or any other kind of PDF, this tool aims to provide valuable insights quickly.

## Tech Stack & Modern AI Techniques

- **Programming Language:** Python
- **Frameworks & Libraries:**
  - Streamlit (for creating interactive web interfaces)
  - Deep learning libraries for natural language processing (e.g., Transformer-based models)
  - PDF processing libraries
  - [LangChain](https://github.com/langchain-ai/langchain) for RAG pipeline
  - [Chroma](https://github.com/chroma-core/chroma) for vector storage

This project utilizes modern AI techniques such as deep learning and transformer models to understand and process natural language from PDF documents. With these techniques, the tool is capable of automating complex workflows and providing intelligent responses to user queries.

## Setup & Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd pdf_ai_agent
   ```

2. **Setup Virtual Environment:**

   It is recommended to use a virtual environment. You can set one up using:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies:**

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   Or, if you use the [uv](https://github.com/astral-sh/uv) package manager:

   ```bash
   uv install
   ```

## Configuration

All configuration is handled via a YAML file (default: `config.yaml`). You can specify a different config file using the `--config` argument.

**Key options:**
- `chunk_size`, `chunk_overlap`: Controls document chunking for embedding.
- `max_tokens`, `temperature`: Model generation parameters.
- `persist_directory`: Where vector DBs are stored.
- `model`: Default model to use (`openai`, `gemini`, `claude`, `ollama`).
- Model/embedding names: Set the specific model names for each provider.

## Running the Project

### Streamlit App

If a web-based interface is desired (using the Streamlit framework), you can run the app by executing:

```bash
streamlit run app.py
```

### Command Line Interface (CLI)

The main script supports two commands: `index` and `ask`. Both require a config file (default: `config.yaml`).

#### Indexing Documents

To index a PDF, DOCX, TXT, or a folder of documents:

```bash
python main.py --config config.yaml index <path_to_file_or_folder> [--metadata <collection_name>]
```

- `<path_to_file_or_folder>`: Path to the file or directory you want to index
- `--metadata <collection_name>`: (Optional) Name for the indexed collection (used as a key for querying)

**Example:**

For a single PDF file

```bash
python main.py --config config.yaml index <file-name.pdf> --metadata deep_learning
```

For a folder containing multiple documents

```bash
python main.py --config config.yaml index <dir/> --metadata deep_learning
```

#### Asking Questions

To ask questions based on the indexed documents:

```bash
python main.py --config config.yaml ask <collection_name> "<your_question>"
```

- `<collection_name>`: The metadata key used during indexing
- `<your_question>`: The question you want to ask

**Example:**

```bash
python main.py --config config.yaml ask deep_learning "What are the main takeaways from this document?"
```

**Notes:**
- Make sure to set the required API keys in your environment before running these commands (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY` as needed).
- You can switch between supported models by changing the `--model` argument or editing the config file.
- The vector database for each collection is stored in the directory specified by `persist_directory` in your config.

## Applications

The IntelliPDF Agent can be applied in various scenarios including:

- **Research:** Automatically summarize and extract key insights from academic papers and research documents.
- **Business:** Analyze financial reports, legal documents, and other business-related PDFs to streamline workflows.
- **Automation:** Integrate with other systems to automate routine document processing tasks, enhancing overall productivity.

## RAG Implementation: Retrieval-Augmented Generation

The Retrieval-Augmented Generation (RAG) technique has been implemented in this project to enhance document processing workflows.

- **Why:** It combines the strengths of retrieval and generation to ensure that AI responses are grounded in the actual document content, reducing inaccuracies.
- **How:** The system first retrieves relevant information from PDFs using vector-based similarity search, then leverages a transformer-based language model to generate contextually accurate outputs.
- **What:** The implementation includes modules for text extraction, vector embedding, similarity search, and integration with AI language models, thereby delivering reliable and context-aware responses.

## Conclusion

This project exemplifies the integration of modern AI techniques with practical application in document processing. It is modular and extensible, making it a valuable starting point for further advancements in automated document analysis and intelligent workflow management.
