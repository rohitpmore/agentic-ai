"""
Embedding utilities for the multi-agent system
"""

from langchain_huggingface import HuggingFaceEmbeddings
from config import config


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Get configured HuggingFace embeddings instance.
    
    Returns:
        HuggingFaceEmbeddings: Configured embeddings model
    """
    return HuggingFaceEmbeddings(model_name=config.embedding_model) 