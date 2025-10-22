import os
import sys
from dotenv import load_dotenv

from src.data_loader import load_adsb_data, preprocess_adsb_data
from src.embed_store import EmbeddingStore
from src.retriever import FlightRetriever
from src.rag_pipeline import ADSBRAGPipeline


def setup_environment():
    """Load environment variables"""
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not found in environment variables!")
        print("\nPlease set your API key in a .env file:")
        print("GROQ_API_KEY=your_api_key_here")
        sys.exit(1)
    
    return groq_api_key


def initialize_vectorstore():
    """Initialize or load the vector store"""
    print("\n" + "=" * 70)
    print("Initializing Vector Store")
    print("=" * 70)
    
    embedding_store = EmbeddingStore(vectorstore_path="data/vectorstore")
    
    # Check if vector store exists
    index, documents, embeddings = embedding_store.load_store()
    
    if index is None:
        print("\nVector store not found. Creating new vector store...\n")
        
        # Load and preprocess data
        adsb_records = load_adsb_data("data/adsb_synthetic.json")
        documents = preprocess_adsb_data(adsb_records)
        
        # Create embeddings
        embeddings = embedding_store.create_embeddings(documents)
        
        # Save vector store
        embedding_store.save_store(documents, embeddings)
    
    return embedding_store


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("ADS-B Flight Data RAG Application")
    print("=" * 70)
    
    # Setup environment
    groq_api_key = setup_environment()
    
    # Initialize vector store
    embedding_store = initialize_vectorstore()
    
    # Create retriever
    print("\nInitializing retriever...")
    retriever = FlightRetriever(embedding_store)
    
    # Create RAG pipeline
    print("Initializing RAG pipeline with Groq LLM...")
    rag_pipeline = ADSBRAGPipeline(
        groq_api_key=groq_api_key,
        retriever=retriever,
        model="openai/gpt-oss-20b"  # Free Groq model
    )
    
    print("✓ RAG Pipeline ready!\n")
    
    # Run interactive chatbot
    rag_pipeline.interactive_chat()


if __name__ == "__main__":
    main()