"""
LLM-based query parser using Gemini API for natural language travel queries
"""

import json
import logging
from typing import Optional
from datetime import datetime
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False
from pydantic import ValidationError

from .query_models import TravelQuery, QueryParsingResult, Currency, TravelPace
from config import config

logger = logging.getLogger(__name__)


class LLMQueryParser:
    """
    LLM-powered travel query parser using Gemini API.
    
    Converts natural language travel queries into structured TravelQuery objects
    using Gemini's language understanding capabilities.
    """
    
    def __init__(self):
        """Initialize the LLM query parser"""
        # Configure Gemini if available
        if GEMINI_AVAILABLE and config.gemini_api_key and config.gemini_api_key != "fallback_gemini_key":
            genai.configure(api_key=config.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.llm_available = True
        else:
            if not GEMINI_AVAILABLE:
                logger.warning("Gemini library not available. Natural language query parsing disabled.")
            else:
                logger.warning("Gemini API key not configured. Natural language query parsing disabled.")
            self.llm_available = False
    
    def parse_query(self, user_query: str) -> QueryParsingResult:
        """
        Parse a natural language travel query into structured data using LLM only.
        
        Args:
            user_query: Natural language travel request
            
        Returns:
            QueryParsingResult with parsed data or error information
        """
        logger.info(f"Parsing query with LLM: {user_query}")
        
        # Check if LLM is available
        if not self.llm_available:
            return QueryParsingResult.error_result(
                "LLM parsing service unavailable. Please configure GEMINI_API_KEY to use natural language queries.",
                user_query
            )
        
        # Parse with LLM only
        try:
            result = self._parse_with_llm(user_query)
            if result.success:
                logger.info(f"LLM parsing successful for destination: {result.query.destination}")
            else:
                logger.error(f"LLM parsing failed: {result.error_message}")
            return result
        except Exception as e:
            logger.error(f"LLM parsing error: {str(e)}")
            return QueryParsingResult.error_result(
                f"LLM parsing failed: {str(e)}", user_query
            )
    
    def _parse_with_llm(self, user_query: str) -> QueryParsingResult:
        """
        Parse query using Gemini LLM with structured output.
        
        Args:
            user_query: Natural language travel request
            
        Returns:
            QueryParsingResult with structured data
        """
        try:
            # Create prompt for structured extraction
            prompt = self._create_extraction_prompt(user_query)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            if not response.text:
                return QueryParsingResult.error_result(
                    "LLM returned empty response", user_query
                )
            
            # Parse JSON response
            try:
                # Extract JSON from response (handle markdown code blocks)
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                parsed_data = json.loads(response_text.strip())
                
                # Create TravelQuery from parsed data
                travel_query = TravelQuery.from_dict(parsed_data)
                
                return QueryParsingResult.success_result(travel_query, user_query)
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse LLM response: {str(e)}")
                logger.debug(f"LLM response was: {response.text}")
                return QueryParsingResult.error_result(
                    f"Invalid LLM response format: {str(e)}", user_query
                )
                
        except Exception as e:
            logger.error(f"LLM API error: {str(e)}")
            return QueryParsingResult.error_result(
                f"LLM API error: {str(e)}", user_query
            )
    
    def _create_extraction_prompt(self, user_query: str) -> str:
        """
        Create prompt for structured data extraction.
        
        Args:
            user_query: User's travel query
            
        Returns:
            Formatted prompt for Gemini
        """
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Get available currencies and paces for the prompt
        currencies = [c.value for c in Currency]
        paces = [p.value for p in TravelPace]
        
        prompt = f"""
Extract travel information from the following natural language query and return it as JSON.

Query: "{user_query}"

Today's date: {today}

Extract the following information and return as valid JSON:

{{
    "destination": "string (required - city or country name)",
    "duration_days": number (optional - trip length in days),
    "budget": number (optional - total budget amount),
    "currency": "string (optional - one of: {', '.join(currencies)})",
    "start_date": "string (optional - YYYY-MM-DD format)",
    "end_date": "string (optional - YYYY-MM-DD format)",
    "group_size": number (optional - number of travelers, default 1),
    "special_requirements": ["string"] (optional - any special needs),
    "preferences": {{
        "pace": "string (optional - one of: {', '.join(paces)})",
        "budget_level": "string (optional - low/medium/high)",
        "accommodation_type": "string (optional - budget/mid-range/luxury)",
        "interests": ["string"] (optional - user interests)
    }}
}}

Rules:
1. ALWAYS include destination (required field)
2. If specific dates aren't mentioned but duration is, leave start_date and end_date empty
3. If no currency is mentioned, use "USD"
4. If no duration is specified, don't include duration_days
5. Extract budget amounts from $ symbols or words like "budget", "spend"
6. Common duration patterns: "3 days", "weekend" (2 days), "week" (7 days)
7. Common date patterns: "next Friday", "in 2 weeks", "January 15th"
8. Return only valid JSON, no explanations

Examples:
- "Trip to Paris for 3 days" → {{"destination": "Paris", "duration_days": 3, "currency": "USD"}}
- "Visit Tokyo with $2000 budget" → {{"destination": "Tokyo", "budget": 2000, "currency": "USD"}}
- "Weekend in London" → {{"destination": "London", "duration_days": 2, "currency": "USD"}}

Extract from: "{user_query}"

JSON:
"""
        return prompt.strip()
    


# Global parser instance
_parser_instance: Optional[LLMQueryParser] = None


def get_query_parser() -> LLMQueryParser:
    """
    Get singleton instance of LLM query parser.
    
    Returns:
        LLMQueryParser instance
    """
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = LLMQueryParser()
    return _parser_instance


def parse_travel_query(user_query: str) -> QueryParsingResult:
    """
    Convenience function to parse a travel query.
    
    Args:
        user_query: Natural language travel request
        
    Returns:
        QueryParsingResult with parsed data
    """
    parser = get_query_parser()
    return parser.parse_query(user_query)