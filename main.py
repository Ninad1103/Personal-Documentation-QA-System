import os
import sys
from app import LocalRAG

def interactive_mode(rag):
    print("\n" + "="*60)
    print(" RAG System Ready! Ask questions about your documents.")
    print("="*60)
    print("\nCommands:")
    print("  - Type your question to get an answer")
    print("  - 'stats' - Show system statistics")
    print("  - 'quit' or 'exit' - Exit the program")
    print("="*60 + "\n")
    
    while True:
        try:
            question = input("\n Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye!")
                break
            
            if question.lower() == 'stats':
                stats = rag.get_stats()
                print("\n System Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            if not question:
                continue
            
            rag.ask(question)
            
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")

def main():
    """Main function"""
    print(" Starting Local RAG System...")
    print("="*60)
    
    doc_files = []
    if os.path.exists("./documents"):
        for ext in ['*.pdf', '*.txt']:
            import glob
            doc_files.extend(glob.glob(f"./documents/**/{ext}", recursive=True))
    
    if not doc_files and not os.path.exists("./chroma_db"):
        print("\n No documents found in the 'documents' folder!")
        print("Please add some PDF or text files before running.")
        return
    
    # Initialize RAG system
    print("\n Initializing RAG system...")
    rag = LocalRAG(
        documents_path="./documents",
        model_name="llama3.2:3b"
    )
    
    if not rag.initialize():
        return
    
    print("\n Initialization complete!")    
    interactive_mode(rag)

if __name__ == "__main__":
    main()