from resources.vector_store_manager import VectorDBManager


class DocumentRetriever:
    """Class for retrieving documents from vector database."""
    
    def __init__(
        self, 
        db_manager: VectorDBManager = None
    ):
        """Initialize with a DB manager or create a new one if not provided."""
        self.db_manager = db_manager if db_manager else VectorDBManager()
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 2, 
        **kwargs
    ):
        """Retrieve documents based on query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters to pass to the retriever
            
        Returns:
            List of retrieved documents
        """
        retriever = self.db_manager.db.as_retriever(
            search_kwargs={"k": top_k, **kwargs}
        )
        return retriever.invoke(query)