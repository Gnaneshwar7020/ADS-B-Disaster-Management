import os
import pickle
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class EmbeddingStore:
    """Create, manage, and persist vector embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vectorstore_path: str = "data/vectorstore"):
        self.model = SentenceTransformer(model_name)
        self.vectorstore_path = vectorstore_path
        self.index = None
        self.documents = None
        
        os.makedirs(vectorstore_path, exist_ok=True)
    
    def create_embeddings(self, documents: List[str]) -> np.ndarray:
        """Generate embeddings for all documents"""
        print(f"Creating embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
        print(f"Embeddings created with shape: {embeddings.shape}")
        return embeddings
    
    def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Build FAISS index from embeddings"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print(f"FAISS index built with {index.ntotal} vectors")
        return index
    
    def save_store(self, documents: List[str], embeddings: np.ndarray):
        """Save embeddings, index, and documents to disk"""
        index_path = os.path.join(self.vectorstore_path, "faiss_index")
        docs_path = os.path.join(self.vectorstore_path, "documents.pkl")
        embeddings_path = os.path.join(self.vectorstore_path, "embeddings.npy")
        
        # Build and save FAISS index
        self.index = self.build_index(embeddings)
        faiss.write_index(self.index, index_path)
        
        # Save documents
        with open(docs_path, 'wb') as f:
            pickle.dump(documents, f)
        
        # Save embeddings
        np.save(embeddings_path, embeddings)
        
        self.documents = documents
        print(f"Vector store saved to {self.vectorstore_path}")
    
    def load_store(self) -> tuple:
        """Load embeddings, index, and documents from disk"""
        index_path = os.path.join(self.vectorstore_path, "faiss_index")
        docs_path = os.path.join(self.vectorstore_path, "documents.pkl")
        embeddings_path = os.path.join(self.vectorstore_path, "embeddings.npy")
        
        if not all(os.path.exists(p) for p in [index_path, docs_path, embeddings_path]):
            print("Vector store not found. Please create it first.")
            return None, None, None
        
        self.index = faiss.read_index(index_path)
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        embeddings = np.load(embeddings_path)
        
        print(f"Vector store loaded from {self.vectorstore_path}")
        return self.index, self.documents, embeddings
    
    def query_store(self, query: str, k: int = 5) -> List[tuple]:
        """Search the vector store and return top-k similar documents"""
        query_embedding = self.model.encode(query, convert_to_numpy=True).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(distance)))
        
        return results