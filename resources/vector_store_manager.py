from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Any, Optional


class VectorDBManager:
    """Singleton class for managing vector database operations."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorDBManager, cls).__new__(cls)
            cls._instance._db = None
        return cls._instance
    
    def initialize_db(
        self, 
        docs: List[Any], 
        embedding_model: Optional[Any] = None
    ):
        """Initialize the vector database with documents.
        
        Args:
            docs: List of documents to index
            embedding_model: Optional custom embedding model, defaults to OpenAIEmbeddings
        """
        if embedding_model is None:
            embedding_model = OpenAIEmbeddings()
            
        self._db = FAISS.from_documents(docs, embedding_model)
        print(f"Database initialized with {self._db.index.ntotal} documents.")
        
    @property
    def db(self):
        """Get the database instance."""
        if self._db is None:
            raise ValueError("Database not initialized. Call initialize_db first.")
        return self._db
    
    @property
    def document_count(self) -> int:
        """Get the number of documents in the database."""
        if self._db is None:
            return 0
        return self._db.index.ntotal
    
