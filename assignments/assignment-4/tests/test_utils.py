"""
Unit tests for utility functions
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent_system.utils.web_search import WebSearchManager
from multi_agent_system.utils.vector_store import VectorStoreManager


class TestWebSearchManager(unittest.TestCase):
    """Test cases for the WebSearchManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.web_search_manager = WebSearchManager(api_key="test_key")
    
    @patch('multi_agent_system.utils.web_search.TavilyClient')
    def test_search_success(self, mock_tavily_client):
        """Test successful web search."""
        # Mock Tavily client response
        mock_response = {
            'answer': 'Test answer',
            'results': [
                {
                    'title': 'Test Title',
                    'url': 'https://example.com',
                    'content': 'Test content here'
                }
            ]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = mock_response
        mock_tavily_client.return_value = mock_client_instance
        
        # Reinitialize to use the mocked client
        self.web_search_manager = WebSearchManager(api_key="test_key")
        self.web_search_manager.client = mock_client_instance
        
        result = self.web_search_manager.search("test query")
        
        self.assertEqual(result, mock_response)
        mock_client_instance.search.assert_called_once()
    
    @patch('multi_agent_system.utils.web_search.TavilyClient')
    def test_search_failure(self, mock_tavily_client):
        """Test web search failure handling."""
        # Mock Tavily client to raise exception
        mock_client_instance = Mock()
        mock_client_instance.search.side_effect = Exception("API Error")
        mock_tavily_client.return_value = mock_client_instance
        
        # Reinitialize to use the mocked client
        self.web_search_manager = WebSearchManager(api_key="test_key")
        self.web_search_manager.client = mock_client_instance
        
        with self.assertRaises(Exception) as context:
            self.web_search_manager.search("test query")
        
        self.assertIn("Web search failed", str(context.exception))
    
    def test_format_search_results(self):
        """Test formatting of search results."""
        mock_response = {
            'answer': 'Test summary answer',
            'results': [
                {
                    'title': 'Test Article',
                    'url': 'https://test.com',
                    'content': 'This is test content for the article'
                }
            ]
        }
        
        formatted = self.web_search_manager.format_search_results("test query", mock_response)
        
        self.assertIn("test query", formatted)
        self.assertIn("Test summary answer", formatted)
        self.assertIn("Test Article", formatted)
        self.assertIn("https://test.com", formatted)
    
    def test_format_search_results_no_answer(self):
        """Test formatting when no answer is provided."""
        mock_response = {
            'results': [
                {
                    'title': 'Test Article',
                    'url': 'https://test.com',
                    'content': 'This is test content'
                }
            ]
        }
        
        formatted = self.web_search_manager.format_search_results("test query", mock_response)
        
        self.assertIn("test query", formatted)
        self.assertIn("Test Article", formatted)
        self.assertNotIn("Summary Answer", formatted)
    
    def test_format_search_results_no_results(self):
        """Test formatting when no results are found."""
        mock_response = {'answer': 'Test answer'}
        
        formatted = self.web_search_manager.format_search_results("test query", mock_response)
        
        self.assertIn("No detailed results found", formatted)
    
    @patch('multi_agent_system.utils.web_search.TavilyClient')
    def test_perform_search_success(self, mock_tavily_client):
        """Test the perform_search method with successful search."""
        mock_response = {
            'answer': 'Test answer',
            'results': [{'title': 'Test', 'url': 'test.com', 'content': 'content'}]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = mock_response
        mock_tavily_client.return_value = mock_client_instance
        
        self.web_search_manager = WebSearchManager(api_key="test_key")
        self.web_search_manager.client = mock_client_instance
        
        result = self.web_search_manager.perform_search("test query")
        
        self.assertIn("test query", result)
        self.assertIn("Test answer", result)
    
    @patch('multi_agent_system.utils.web_search.TavilyClient')
    def test_perform_search_failure(self, mock_tavily_client):
        """Test the perform_search method with search failure."""
        mock_client_instance = Mock()
        mock_client_instance.search.side_effect = Exception("API Error")
        mock_tavily_client.return_value = mock_client_instance
        
        self.web_search_manager = WebSearchManager(api_key="test_key")
        self.web_search_manager.client = mock_client_instance
        
        result = self.web_search_manager.perform_search("test query")
        
        self.assertIn("Unable to perform web search", result)
        self.assertIn("test query", result)
    
    @patch('multi_agent_system.utils.web_search.TavilyClient')
    def test_test_connection_success(self, mock_tavily_client):
        """Test successful connection test."""
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = {'results': []}
        mock_tavily_client.return_value = mock_client_instance
        
        self.web_search_manager = WebSearchManager(api_key="test_key")
        self.web_search_manager.client = mock_client_instance
        
        result = self.web_search_manager.test_connection()
        
        self.assertTrue(result)
    
    @patch('multi_agent_system.utils.web_search.TavilyClient')
    def test_test_connection_failure(self, mock_tavily_client):
        """Test failed connection test."""
        mock_client_instance = Mock()
        mock_client_instance.search.side_effect = Exception("Connection failed")
        mock_tavily_client.return_value = mock_client_instance
        
        self.web_search_manager = WebSearchManager(api_key="test_key")
        self.web_search_manager.client = mock_client_instance
        
        result = self.web_search_manager.test_connection()
        
        self.assertFalse(result)


class TestVectorStoreManager(unittest.TestCase):
    """Test cases for the VectorStoreManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.faiss_dir = os.path.join(self.temp_dir, "faiss")
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.faiss_dir, exist_ok=True)
        
        # Create a test text file
        self.test_file_path = os.path.join(self.data_dir, "test.txt")
        with open(self.test_file_path, 'w') as f:
            f.write("This is test content for the vector store.\nIt has multiple lines.\nFor testing purposes.")
        
        # Mock embeddings to avoid loading actual model
        with patch('multi_agent_system.utils.vector_store.get_embeddings'):
            self.vector_store_manager = VectorStoreManager(
                data_directory=self.data_dir,
                faiss_directory=self.faiss_dir
            )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_documents_success(self):
        """Test successful document loading."""
        documents = self.vector_store_manager.load_documents()
        
        self.assertEqual(len(documents), 1)
        self.assertIn("test content", documents[0].page_content)
    
    def test_load_documents_no_directory(self):
        """Test document loading with non-existent directory."""
        vector_store_manager = VectorStoreManager(
            data_directory="/nonexistent/path",
            faiss_directory=self.faiss_dir
        )
        
        with self.assertRaises(FileNotFoundError):
            vector_store_manager.load_documents()
    
    def test_create_chunks(self):
        """Test document chunking."""
        documents = self.vector_store_manager.load_documents()
        chunks = self.vector_store_manager.create_chunks(documents)
        
        self.assertGreater(len(chunks), 0)
        # Should create multiple chunks due to line breaks
        self.assertGreaterEqual(len(chunks), 1)
    
    def test_format_docs(self):
        """Test document formatting."""
        # Create mock documents
        mock_docs = [
            Mock(page_content="First document content"),
            Mock(page_content="Second document content")
        ]
        
        formatted = VectorStoreManager.format_docs(mock_docs)
        
        self.assertIn("First document content", formatted)
        self.assertIn("Second document content", formatted)
        self.assertIn("\n\n", formatted)  # Documents should be separated
    
    @patch('multi_agent_system.utils.vector_store.FAISS.from_documents')
    def test_create_vector_store_new(self, mock_faiss_from_docs):
        """Test creation of new vector store."""
        # Mock FAISS vector store
        mock_vector_store = Mock()
        mock_faiss_from_docs.return_value = mock_vector_store
        
        vector_store = self.vector_store_manager.create_vector_store(force_recreate=True)
        
        self.assertEqual(vector_store, mock_vector_store)
        mock_faiss_from_docs.assert_called_once()
        mock_vector_store.save_local.assert_called_once_with(self.faiss_dir)
    
    @patch('multi_agent_system.utils.vector_store.FAISS.load_local')
    def test_create_vector_store_load_existing(self, mock_faiss_load):
        """Test loading of existing vector store."""
        # Create a fake index file
        index_file = os.path.join(self.faiss_dir, "index.faiss")
        open(index_file, 'a').close()
        
        mock_vector_store = Mock()
        mock_faiss_load.return_value = mock_vector_store
        
        vector_store = self.vector_store_manager.create_vector_store(force_recreate=False)
        
        self.assertEqual(vector_store, mock_vector_store)
        mock_faiss_load.assert_called_once()
    
    @patch('multi_agent_system.utils.vector_store.FAISS.from_documents')
    def test_get_retriever(self, mock_faiss_from_docs):
        """Test retriever creation."""
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_faiss_from_docs.return_value = mock_vector_store
        
        retriever = self.vector_store_manager.get_retriever()
        
        self.assertEqual(retriever, mock_retriever)
        mock_vector_store.as_retriever.assert_called_once()


if __name__ == '__main__':
    unittest.main() 