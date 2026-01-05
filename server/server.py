from fastapi import FastAPI, HTTPException ,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app import LocalRAG
import os
from typing import Optional, List
import asyncio
import json
from pathlib import Path
import shutil

app = FastAPI(title = "RAG System API", description = "RAG System API")

#cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#initialize the local rag
rag = None
DOCUMENTS_PATH = "./documents"
Path(DOCUMENTS_PATH).mkdir(exist_ok=True)

class Question(BaseModel):
    question: str
    stream: Optional[bool] = False

class InitResponse(BaseModel):
    message: str
    status: str
    stats: Optional[dict] = None

@app.on_event("startup")
async def startup_event():
    global rag
    rag = LocalRAG(DOCUMENTS_PATH, model_name = "llama3.2:3b")
    print("RAG initialized (lazy loading)")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "rag_initialized": rag is not None and rag.vectorstore is not None
    }
        
"""
Endpoints needed:
    1. /health - get
    2. /upload - post
    3. /stats   - get
    4. /query   - post
    5. /reinitialize - post 
"""

@app.post("/upload")
async def upload_documents(file: UploadFile = File(...)):
    """Upload documents to the RAG system"""
    if not file.filename.endswith(('.pdf','.txt')):
        raise HTTPException(status_code=400, detail="Only pdf and text files are allowed")
    
    try:
        file_path = Path(DOCUMENTS_PATH) / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        #reinitialize the rag
        global rag
        rag = LocalRAG(DOCUMENTS_PATH, model_name = "llama3.2:3b")
        #Force rebuild of vectorstore to remove the existing one
        if Path("./chroma_db").exists():
            shutil.rmtree("./chroma_db")
        
        success = rag.initialize()
        if success:
            return {
                    "status" : "success",
                    "message" : f"File '{file.filename}' uploaded and RAG system reinitialized successfully",
                    "filename" : file.filename
                    }
        else:
            raise HTTPException(status_code=500, detail="Failed to process document")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system stats"""
    if not rag:
        return {"status": "Not initialized", "total_documents": 0, "total_chunks": 0}
    stats = rag.get_stats()
    if isinstance(stats, str):
        return {"status": "Inactive", "total_documents": 0, "total_chunks": 0}
    return {"status": "Active", **stats}

@app.post("/query", response_model=dict)
async def query(question: Question):
    """Query the RAG system"""
    if not rag or not rag.qa_chain:
        raise HTTPException(status_code=400, detail="RAG system is not initialized")
    #question cannot be empty
    if not question.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        print(f"Received question: {question.question}")
        result = rag.ask(question.question, verbose = False)
        print(f"RAG result keys: {result.keys()}")
        sources = [doc.page_content[:200] for doc in result.get('source_documents', [])]
        return {
            "answer": result['result'],
            "sources": sources
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files from rag_app folder
app.mount("/static", StaticFiles(directory="rag_app"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("rag_app/index.html")

@app.post("/delete")
async def delete_documents():
    """Delete all documents and reset the system"""
    try:
        # Delete files in documents folder
        doc_path = Path(DOCUMENTS_PATH)
        for file in doc_path.glob("**/*"):
            if file.is_file():
                file.unlink()
        
        # Delete vectorstore
        if Path("./chroma_db").exists():
            shutil.rmtree("./chroma_db")
            
        # Reset RAG instance
        global rag
        rag = LocalRAG(DOCUMENTS_PATH, model_name = "llama3.2:3b")
        
        return {"message": "All documents deleted and system reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
