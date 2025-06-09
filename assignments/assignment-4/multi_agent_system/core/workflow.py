"""
Main workflow orchestration for the multi-agent system
"""

from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from .state import AgentState
from ..agents.supervisor import SupervisorAgent
from ..agents.rag_agent import RAGAgent
from ..agents.llm_agent import LLMAgent
from ..agents.web_crawler_agent import WebCrawlerAgent
from ..agents.validation_agent import ValidationAgent
from ..utils.vector_store import VectorStoreManager
from ..utils.web_search import WebSearchManager
from config import config


class MultiAgentWorkflow:
    """
    Main workflow class that orchestrates the multi-agent system.
    
    This class creates and manages all agents, sets up the workflow graph,
    and provides interfaces for executing queries.
    """
    
    def __init__(self):
        """Initialize the multi-agent workflow system."""
        self.model = self._create_model()
        self.vector_store_manager = VectorStoreManager()
        self.web_search_manager = WebSearchManager()
        
        # Initialize all agents
        self.supervisor_agent = SupervisorAgent(self.model)
        self.rag_agent = RAGAgent(self.model, self.vector_store_manager)
        self.llm_agent = LLMAgent(self.model)
        self.web_crawler_agent = WebCrawlerAgent(self.model, self.web_search_manager)
        self.validation_agent = ValidationAgent(self.model)
        
        # Create and compile the workflow
        self.app = self._create_workflow()
    
    def _create_model(self) -> ChatGoogleGenerativeAI:
        """
        Create and configure the Google Gemini model.
        
        Returns:
            ChatGoogleGenerativeAI: Configured model instance
        """
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            google_api_key=config.gemini_api_key
        )
    
    def _create_workflow(self):
        """
        Create and configure the workflow graph.
        
        Returns:
            Compiled workflow application
        """
        # Create the workflow
        workflow = StateGraph(AgentState)
        
        # Add all nodes to the workflow
        workflow.add_node("Supervisor", self.supervisor_agent.process)
        workflow.add_node("RAG", self.rag_agent.process)
        workflow.add_node("LLM", self.llm_agent.process)
        workflow.add_node("Web Crawler", self.web_crawler_agent.process)
        workflow.add_node("Validation", self.validation_agent.process)
        
        # Set entry point
        workflow.set_entry_point("Supervisor")
        
        # Add conditional routing from Supervisor
        workflow.add_conditional_edges(
            "Supervisor",
            self.supervisor_agent.route_decision,
            {
                "Call RAG": "RAG",
                "Call LLM": "LLM", 
                "Call Web Crawler": "Web Crawler",
            }
        )
        
        # All specialized nodes go to validation
        workflow.add_edge("RAG", "Validation")
        workflow.add_edge("LLM", "Validation")
        workflow.add_edge("Web Crawler", "Validation")
        
        # FEEDBACK LOOP: Validation can route back to Supervisor or end
        workflow.add_conditional_edges(
            "Validation",
            self.validation_agent.feedback_router,
            {
                "END": END,
                "RETRY": "Supervisor"  # This creates the feedback loop!
            }
        )
        
        # Compile and return the workflow
        return workflow.compile()
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a user query through the multi-agent system.
        
        Args:
            question: The user's question
            
        Returns:
            Dict[str, Any]: Complete workflow result
        """
        # Validate query format
        is_valid, validation_msg = self.validation_agent.validate_query_format(question)
        if not is_valid:
            return {
                "question": question,
                "answer": f"Invalid query: {validation_msg}",
                "status": "error",
                "validation_status": "FAILED",
                "retry_count": 0
            }
        
        # Reset retry count for new query
        self.validation_agent.reset_retry_count()
        
        try:
            # Execute the workflow
            state = {"messages": [question]}
            result = self.app.invoke(state)
            
            # Extract and format the result
            return self._format_result(question, result)
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "question": question,
                "answer": f"I'm sorry, there was a technical issue processing your question: {error_msg}",
                "status": "error",
                "validation_status": "ERROR",
                "retry_count": self.validation_agent.get_retry_count()
            }
    
    def _format_result(self, question: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the workflow result for presentation.
        
        Args:
            question: Original question
            result: Raw workflow result
            
        Returns:
            Dict[str, Any]: Formatted result
        """
        # Extract the final response and validation status
        final_response = None
        validation_status = None
        
        for msg in result["messages"]:
            if "VALIDATION_PASS" in str(msg):
                validation_status = "PASSED"
            elif "VALIDATION_FAIL" in str(msg):
                validation_status = "FAILED"
            elif str(msg) not in [question] and "VALIDATION" not in str(msg):
                final_response = msg
        
        # Handle case where no final response was found
        if final_response is None:
            final_response = "No response generated"
            validation_status = validation_status or "UNKNOWN"
        
        return {
            "question": question,
            "answer": str(final_response),
            "status": "success" if validation_status == "PASSED" else "partial",
            "validation_status": validation_status or "UNKNOWN",
            "retry_count": self.validation_agent.get_retry_count(),
            "full_result": result
        }
    
    def test_system(self) -> Dict[str, Any]:
        """
        Run a comprehensive test of the system.
        
        Returns:
            Dict[str, Any]: Test results
        """
        test_queries = [
            ("What is the structure of the US economy?", "USA Economy"),
            ("Explain how photosynthesis works", "General Knowledge"),
            ("Latest AI news today", "Real-time/Current Events"),
        ]
        
        results = {}
        print("🧪 Testing Multi-Agent System")
        print("=" * 60)
        
        for query, expected_category in test_queries:
            print(f"\n🔸 Testing: {query}")
            print(f"🎯 Expected Category: {expected_category}")
            print("-" * 40)
            
            try:
                result = self.query(query)
                results[query] = result
                
                print(f"📤 Status: {result['status']}")
                print(f"✅ Validation: {result['validation_status']}")
                print(f"🔄 Retries: {result['retry_count']}")
                print(f"📝 Answer: {result['answer'][:200]}...")
                
            except Exception as e:
                print(f"❌ Test failed: {str(e)}")
                results[query] = {"error": str(e)}
            
            print("-" * 40)
        
        return results
    
    def initialize_vector_store(self, force_recreate: bool = False):
        """
        Initialize the vector store.
        
        Args:
            force_recreate: Whether to force recreation of the vector store
        """
        self.vector_store_manager.create_vector_store(force_recreate=force_recreate)
    
    def test_connections(self) -> Dict[str, bool]:
        """
        Test all external connections.
        
        Returns:
            Dict[str, bool]: Connection test results
        """
        print("🔌 Testing External Connections")
        print("-" * 30)
        
        results = {}
        
        # Test Gemini API
        try:
            test_response = self.model.invoke("Hello")
            results["gemini_api"] = True
            print("✅ Gemini API: Connected")
        except Exception as e:
            results["gemini_api"] = False
            print(f"❌ Gemini API: Failed - {str(e)}")
        
        # Test Tavily API
        results["tavily_api"] = self.web_search_manager.test_connection()
        
        # Test Vector Store
        try:
            self.vector_store_manager.create_vector_store()
            results["vector_store"] = True
            print("✅ Vector Store: Ready")
        except Exception as e:
            results["vector_store"] = False
            print(f"❌ Vector Store: Failed - {str(e)}")
        
        return results 