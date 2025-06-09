"""
RAG agent for USA Economy questions using vector search
"""

from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ..core.state import AgentState
from ..utils.vector_store import VectorStoreManager
from config import config


class RAGAgent:
    """
    RAG (Retrieval-Augmented Generation) agent for USA Economy questions.
    
    This agent specializes in answering questions about the US economy using
    a vector database of relevant documents for context retrieval.
    """
    
    def __init__(self, model: ChatGoogleGenerativeAI, vector_store_manager: VectorStoreManager):
        """
        Initialize the RAG agent.
        
        Args:
            model: Google Gemini model instance
            vector_store_manager: Vector store manager for document retrieval
        """
        self.model = model
        self.vector_store_manager = vector_store_manager
        self.retriever = self.vector_store_manager.get_retriever()
        
        self.prompt_template = PromptTemplate(
            template="""You are an assistant specializing in US Economy questions. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:""",
            input_variables=['context', 'question']
        )
        
        self.output_parser = StrOutputParser()
        
        # Create the RAG chain
        self.rag_chain = (
            {"context": self.retriever | self.vector_store_manager.format_docs, "question": RunnablePassthrough()}
            | self.prompt_template 
            | self.model 
            | self.output_parser
        )
    
    def answer_question(self, question: str) -> str:
        """
        Answer a USA Economy question using RAG.
        
        Args:
            question: The question to answer
            
        Returns:
            str: The generated answer
        """
        try:
            print(f"🏛️ RAG Node (USA Economy) processing: {question}")
            response = self.rag_chain.invoke(question)
            print("✅ RAG Response generated")
            return response
        except Exception as e:
            error_msg = f"RAG processing failed: {str(e)}"
            print(f"❌ RAG Error: {error_msg}")
            return f"I'm sorry, I couldn't process your USA Economy question due to a technical issue: {error_msg}"
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process the agent state and generate a response.
        
        Args:
            state: Current agent state
            
        Returns:
            Dict[str, Any]: Updated state with response
        """
        question = state["messages"][0]
        response = self.answer_question(str(question))
        return {"messages": [response]}
    
    def get_relevant_documents(self, question: str, k: int = 4):
        """
        Retrieve relevant documents for a question.
        
        Args:
            question: The question to find documents for
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            return self.retriever.get_relevant_documents(question)
        except Exception as e:
            print(f"❌ Document retrieval error: {e}")
            return [] 