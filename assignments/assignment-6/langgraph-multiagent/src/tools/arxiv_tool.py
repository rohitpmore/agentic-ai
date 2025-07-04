import arxiv
from typing import List, Dict, Any, Optional
import logging
import time
from datetime import datetime


class ArxivTool:
    """Tool for searching arXiv papers"""
    
    def __init__(self, max_results: int = 10, sort_by: str = "relevance"):
        self.max_results = max_results
        self.sort_by = sort_by
        self.logger = logging.getLogger("arxiv_tool")
        
    def search_papers(
        self,
        query: str,
        category: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search arXiv papers by query"""
        
        try:
            # Prepare search query
            search_query = query
            if category:
                search_query = f"cat:{category} AND {query}"
            
            # Configure search
            search = arxiv.Search(
                query=search_query,
                max_results=max_results or self.max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            # Execute search
            results = []
            for result in search.results():
                paper_data = {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "url": result.entry_id,
                    "published": result.published.isoformat(),
                    "categories": result.categories,
                    "relevance_score": self._calculate_relevance(result, query)
                }
                results.append(paper_data)
            
            self.logger.info(f"Found {len(results)} papers for query: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"ArXiv search failed: {e}")
            return []
    
    def _calculate_relevance(self, paper: arxiv.Result, query: str) -> float:
        """Calculate relevance score for paper"""
        
        score = 0.0
        query_terms = query.lower().split()
        
        # Title relevance (50%)
        title_lower = paper.title.lower()
        title_matches = sum(1 for term in query_terms if term in title_lower)
        title_score = title_matches / len(query_terms) if query_terms else 0
        score += 0.5 * title_score
        
        # Abstract relevance (40%)
        abstract_lower = paper.summary.lower()
        abstract_matches = sum(1 for term in query_terms if term in abstract_lower)
        abstract_score = abstract_matches / len(query_terms) if query_terms else 0
        score += 0.4 * abstract_score
        
        # Recency bonus (10%)
        days_since_publication = (datetime.now() - paper.published).days
        recency_score = max(0, 1 - (days_since_publication / 365))
        score += 0.1 * recency_score
        
        return min(score, 1.0)