"""
Supervisor agent for query classification and orchestration
"""

from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..core.state import AgentState
from ..core.parsers import get_topic_parser
from config import config


class SupervisorAgent:
    """
    Supervisor agent that classifies queries and orchestrates the workflow.
    
    The supervisor analyzes incoming queries and classifies them into one of three categories:
    - USA Economy: Questions about US economic structure, policies, GDP, etc.
    - General Knowledge: General questions that don't require real-time data
    - Real-time/Current Events: Questions about latest news, current prices, etc.
    """
    
    def __init__(self, model: ChatGoogleGenerativeAI):
        """
        Initialize the supervisor agent.
        
        Args:
            model: Google Gemini model instance
        """
        self.model = model
        self.parser = get_topic_parser()
        self.prompt_template = PromptTemplate(
            template="""
            Your task is to classify the question into one of the following categories: [USA Economy, General Knowledge, Real-time/Current Events]
            
            Guidelines:
            - USA Economy: Questions about US economic structure, policies, GDP, Federal Reserve, trade, financial systems
            - General Knowledge: General questions, explanations, concepts that don't require real-time data
            - Real-time/Current Events: Questions about latest news, current prices, today's weather, recent developments
            
            User Query: {question}
            {format_instructions}
            """,
            input_variables=["question"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = self.prompt_template | self.model | self.parser
    
    def classify_query(self, query: str) -> Dict[str, str]:
        """
        Classify a user query into appropriate category.
        
        Args:
            query: User query string
            
        Returns:
            Dict[str, str]: Classification result with topic and reasoning
        """
        try:
            result = self.chain.invoke({"question": query})
            print(f"🔍 Supervisor analyzing question: {query}")
            print(f"📊 Classification result: {result.Topic}")
            print(f"💭 Reasoning: {result.Reasoning}")
            
            return {
                "topic": result.Topic,
                "reasoning": result.Reasoning
            }
        except Exception as e:
            print(f"❌ Supervisor classification error: {e}")
            # Fallback to general knowledge for unknown cases
            return {
                "topic": "General Knowledge",
                "reasoning": f"Classification failed, defaulting to general knowledge: {str(e)}"
            }
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process the agent state and return classification.
        
        Args:
            state: Current agent state
            
        Returns:
            Dict[str, Any]: Updated state with classification
        """
        question = state['messages'][-1]
        classification = self.classify_query(str(question))
        
        return {"messages": [classification["topic"]]}
    
    def route_decision(self, state: AgentState) -> str:
        """
        Make routing decision based on classification.
        
        Args:
            state: Current agent state
            
        Returns:
            str: Routing decision
        """
        classification = state["messages"][-1]
        print(f"🔀 Router decision based on: {classification}")

        if "USA Economy" in str(classification):
            print("→ Routing to RAG Node")
            return "Call RAG"
        elif "General Knowledge" in str(classification):
            print("→ Routing to LLM Node") 
            return "Call LLM"
        elif "Real-time/Current Events" in str(classification):
            print("→ Routing to Web Crawler Node")
            return "Call Web Crawler"
        else:
            # Fallback to LLM for unknown classifications
            print("→ Fallback routing to LLM Node")
            return "Call LLM" 