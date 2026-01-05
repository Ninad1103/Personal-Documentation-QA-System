## Local RAG System

A personal Documention Question Answering System. 

## Features

- PDF and Text file processing
- Document chunking
- Vector store creation
- Question answering with source documents
- Statistics about loaded documents

## Installation
Please make sure to have ollama installed and running.
Make sure to be in the RAG directory.

```bash
pip install -r requirements.txt
```

## Usage
Server:
```bash
python.exe -m uvicorn server.server:app --reload
```

Client:
```bash
python -m http.server 8080
```

For terminal based usage:
```bash
python main.py
```