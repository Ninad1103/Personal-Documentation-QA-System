from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import os

class LocalRAG:
    def __init__(self, documents_path="./documents", model_name="llama3.2:3b"):
        self.documents_path = documents_path
        self.model_name = model_name
        self.vectorstore = None
        self.qa_chain = None
    
    def load_documents(self):
        pdf_loader = DirectoryLoader(
            self.documents_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        )
        
        txt_loader = DirectoryLoader(
            self.documents_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
        )
        
        documents = pdf_loader.load() + txt_loader.load()
        print(f"Loaded {len(documents)} documents.")
        return documents

    def split_documents(self, documents):
        print(f"Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")
        return chunks

    def create_vectorstore(self, chunks):
        print(f"Creating vectorstore...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        self.vectorstore = Chroma.from_documents(chunks, embeddings)
        print(f"Vectorstore created.")
    
    def create_qa_chain(self):
        print(f"Creating QA chain...")
        llm = Ollama(model=self.model_name)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate.from_template(
                    """
                    <s>[INST] You are a helpful assistant. 
                    Given the following extracted parts of a long document and a question, 
                    answer the question.
                    If you don't know the answer, just say that you don't know.
                    <</s>
                    <s>[INST] {context}
                    Question: {question}
                    Answer: 
                    <</s>
                    """
                )
            }
        )
        print(f"QA chain created.")

    def initialize(self):
        if os.path.exists("./chroma_db"):
            print("Loading existing vector store...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            self.vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
        else:
            documents = self.load_documents()
            if not documents:
                print("No documents found! Please add PDFs or text files to the 'documents' folder.")
                return False
            
            chunks = self.split_documents(documents)
            self.create_vectorstore(chunks)
        
        self.create_qa_chain()
        return True

    def ask(self, question, verbose=True):
        if not self.qa_chain:
            raise Exception("System not initialized! Call initialize() first.")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"{'='*60}")
            print("Thinking...")
        
        result = self.qa_chain.invoke({"query": question})
        
        if verbose:
            print(f"\nAnswer: {result['result']}")
            print(f"\n📚 Sources: {len(result['source_documents'])} document chunks used")
            
            # Optionally show source
            if result['source_documents']:
                print("\nRelevant excerpts:")
                for i, doc in enumerate(result['source_documents'][:2], 1):
                    content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    print(f"\n{i}. {content}")
        
        return result
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"{'='*60}")
            print("Thinking...")

        result = self.qa_chain.invoke({"query": question})
        
        if verbose:
            print(f"\nAnswer: {result['result']}")
            print(f"\n📚 Sources: {len(result['source_documents'])} document chunks used")
            
            # Optionally show source 
            if result['source_documents']:
                print("\nRelevant excerpts:")
                for i, doc in enumerate(result['source_documents'][:2], 1):
                    content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    print(f"\n{i}. {content}")
        
        return result
    
    def get_stats(self):
        if not self.vectorstore:
            return "No vector store loaded"
        
        collection = self.vectorstore._collection
        count = collection.count()
        
        return {
            "total_chunks": count,
            "model": self.model_name,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }       