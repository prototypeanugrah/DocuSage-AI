import os
import shutil
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
from dotenv import load_dotenv

from backend.config import LLMConfig
from backend.services.rag_service import RAGService

app = FastAPI(title="DocuSage AI API")

# CORS configuration
origins = [
    "http://localhost:5173",  # Vite default port
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Service
# We initialize it once at startup
load_dotenv()
config = LLMConfig()
rag_service = RAGService(config)


class ChatRequest(BaseModel):
    collection_name: str
    question: str
    k: int = 6


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    focus_source: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "DocuSage AI API is running"}


@app.get("/collections")
async def list_collections():
    """List available collections in the persist directory."""
    persist_dir = rag_service.persist_directory
    print(f"Looking for collections in: {persist_dir}")
    
    if not os.path.exists(persist_dir):
        print(f"Directory not found: {persist_dir}")
        return {"collections": []}
    
    collections = [
        d for d in os.listdir(persist_dir) 
        if os.path.isdir(os.path.join(persist_dir, d))
    ]
    print(f"Found collections: {collections}")
    return {"collections": collections}


@app.post("/index")
async def index_documents(
    collection_name: str = Form(...), files: List[UploadFile] = File(...)
):
    """Upload PDFs and index them into a collection."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Create a temporary directory to process files
    with tempfile.TemporaryDirectory() as temp_dir:
        saved_files = []
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                continue

            file_path = os.path.join(temp_dir, file.filename)
            try:
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                saved_files.append(file_path)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save file {file.filename}: {str(e)}",
                )

        if not saved_files:
            raise HTTPException(status_code=400, detail="No valid PDF files found")

        try:
            # Index the directory
            rag_service.index(data_path=temp_dir, collection_name=collection_name)
            return {
                "message": f"Successfully indexed {len(saved_files)} files into collection '{collection_name}'",
                "collection": collection_name,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Query a collection."""
    try:
        response = rag_service.query(
            collection_name=request.collection_name,
            question=request.question,
            k=request.k,
        )
        if response is None:
            raise HTTPException(status_code=404, detail="No response returned")

        # Transform sources to be JSON serializable
        serializable_sources = []
        for doc in response["sources"]:
            serializable_sources.append(
                {"page_content": doc.page_content, "metadata": doc.metadata}
            )

        return {
            "answer": response["answer"],
            "sources": serializable_sources,
            "focus_source": response["focus_source"],
        }
    except ValueError as ve:
        traceback.print_exc()
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
