from typing import Any, Dict, List
from langchain_core.tools import tool
from resources.retriever import DocumentRetriever
from resources.vector_store_manager import VectorDBManager


@tool
def allergy_retriever(
    user_message: str,
) -> List[Dict[str, Any]]:
    """Retrieve documents related to ALLERGY from the vector store based on the original user message"""
    db_manager = VectorDBManager()
    retriever = DocumentRetriever(db_manager)
    results = retriever.retrieve(user_message)
    chunks_result = "\n--- <chunk/> ---\n".join([doc.page_content for doc in results])
    return chunks_result


@tool
def digestive_retriever(
    user_message: str,
) -> List[Dict[str, Any]]:
    """Retrieve documents related to DIGESTIVE from the vector store based on the original user message"""
    db_manager = VectorDBManager()
    retriever = DocumentRetriever(db_manager)
    results = retriever.retrieve(user_message)
    chunks_result = "\n--- <chunk/> ---\n".join([doc.page_content for doc in results])
    return chunks_result


@tool
def vision_loss_retriever(
    user_message: str,
) -> List[Dict[str, Any]]:
    """Retrieve documents related to VISION LOSS from the vector store based on the original user message"""
    db_manager = VectorDBManager()
    retriever = DocumentRetriever(db_manager)
    results = retriever.retrieve(user_message)
    chunks_result = "\n--- <chunk/> ---\n".join([doc.page_content for doc in results])
    return chunks_result


# collected tools
allergy_tools = [
    allergy_retriever,
]

digestive_tools = [
    digestive_retriever,
]

vision_loss_tools = [
    vision_loss_retriever
]