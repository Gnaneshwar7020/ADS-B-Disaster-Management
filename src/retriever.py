from typing import List, Tuple
from src.embed_store import EmbeddingStore


class FlightRetriever:
    """Retrieve relevant flight data based on queries"""
    
    def __init__(self, embedding_store: EmbeddingStore):
        self.embedding_store = embedding_store
    
    def retrieve_context(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve top-k most relevant flight documents for a query
        
        Args:
            query: User query about flights
            k: Number of documents to retrieve
            
        Returns:
            List of relevant flight documents
        """
        results = self.embedding_store.query_store(query, k=k)
        
        # Extract just the documents, discard distances
        documents = [doc for doc, _ in results]
        
        return documents
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents with relevance scores
        
        Args:
            query: User query about flights
            k: Number of documents to retrieve
            
        Returns:
            List of tuples (document, relevance_score)
        """
        return self.embedding_store.query_store(query, k=k)
    
    def format_context(self, documents: List[str]) -> str:
        """
        Format retrieved documents into a context string for the LLM
        
        Args:
            documents: List of retrieved flight documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant flight data found."
        
        context = "Retrieved Flight Data:\n" + "=" * 50 + "\n\n"
        for i, doc in enumerate(documents, 1):
            context += f"Document {i}:\n{doc}\n\n"
        
        return context
    
    def get_augmented_context(self, query: str, k: int = 5) -> str:
        """
        Get formatted context for RAG pipeline
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            Formatted context string ready for LLM
        """
        documents = self.retrieve_context(query, k=k)
        return self.format_context(documents)