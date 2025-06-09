"""
Vector store management utilities for FAISS database
"""

import os
from typing import List, Optional
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from .embeddings import get_embeddings
from config import config


class VectorStoreManager:
    """
    Manages FAISS vector store operations including creation and retrieval.
    """
    
    def __init__(self, data_directory: Optional[str] = None, faiss_directory: Optional[str] = None):
        """
        Initialize the vector store manager.
        
        Args:
            data_directory: Directory containing text files for the knowledge base
            faiss_directory: Directory to store/load FAISS index
        """
        self.data_directory = data_directory or config.data_directory
        self.faiss_directory = faiss_directory or config.faiss_db_directory
        self.embeddings = get_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self._vector_store: Optional[FAISS] = None
    
    def load_documents(self) -> List[Document]:
        """
        Load documents from the data directory.
        
        Returns:
            List[Document]: Loaded documents
            
        Raises:
            FileNotFoundError: If data directory doesn't exist
        """
        if not os.path.exists(self.data_directory):
            raise FileNotFoundError(f"Data directory not found: {self.data_directory}")
        
        document_loader = DirectoryLoader(
            path=self.data_directory,
            glob="./*.txt",
            loader_cls=TextLoader
        )
        
        raw_documents = document_loader.load()
        print(f"Loaded {len(raw_documents)} documents from {self.data_directory}")
        return raw_documents
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: Raw documents to split
            
        Returns:
            List[Document]: Document chunks
        """
        chunks = self.text_splitter.split_documents(documents=documents)
        print(f"Created {len(chunks)} document chunks")
        return chunks
    
    def create_vector_store(self, force_recreate: bool = False) -> FAISS:
        """
        Create or load FAISS vector store.
        
        Args:
            force_recreate: Whether to force recreation even if index exists
            
        Returns:
            FAISS: Vector store instance
        """
        index_path = os.path.join(self.faiss_directory, "index.faiss")
        
        # Try to load existing index if not forcing recreation
        if not force_recreate and os.path.exists(index_path):
            try:
                print(f"Loading existing FAISS index from {self.faiss_directory}")
                self._vector_store = FAISS.load_local(
                    self.faiss_directory, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return self._vector_store
            except Exception as e:
                print(f"Failed to load existing index: {e}")
                print("Creating new index...")
        
        # Create new index
        documents = self.load_documents()
        chunks = self.create_chunks(documents)
        
        if not chunks:
            raise ValueError("No document chunks created. Check your data directory.")
        
        self._vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # Save the index
        os.makedirs(self.faiss_directory, exist_ok=True)
        self._vector_store.save_local(self.faiss_directory)
        print(f"FAISS vector database created and saved with {len(chunks)} documents")
        
        return self._vector_store
    
    def get_retriever(self, **kwargs) -> VectorStoreRetriever:
        """
        Get a retriever from the vector store.
        
        Args:
            **kwargs: Additional arguments for the retriever
            
        Returns:
            VectorStoreRetriever: Configured retriever
        """
        if self._vector_store is None:
            self._vector_store = self.create_vector_store()
        
        return self._vector_store.as_retriever(**kwargs)
    
    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """
        Format retrieved documents for context.
        
        Args:
            docs: Retrieved documents
            
        Returns:
            str: Formatted document content
        """
        return "\n\n".join(doc.page_content for doc in docs) 