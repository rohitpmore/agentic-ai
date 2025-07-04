from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.tools.arxiv_tool import ArxivTool
from src.utils.error_handling import ErrorHandler
from config.settings import Settings


class FinancialResearcher(BaseAgent):
    """Financial and economic researcher agent"""
    
    def __init__(self, settings: Settings):
        super().__init__("financial_researcher")
        self.settings = settings
        self.model = ChatOpenAI(
            model=settings.researcher_model,
            temperature=0.1,
            timeout=settings.timeout_seconds
        )
        self.arxiv_tool = ArxivTool()
        self.error_handler = ErrorHandler(max_retries=settings.max_retries)
        
        # Financial research specializations
        self.specializations = [
            "market_analysis",
            "investment_strategies",
            "risk_assessment",
            "economic_indicators",
            "financial_modeling",
            "algorithmic_trading"
        ]
        
    def get_required_fields(self) -> List[str]:
        return ["research_topic", "research_status"]
        
    def process(self, state: Dict[str, Any]) -> Command:
        """Process financial research request"""
        
        start_time = datetime.now()
        
        try:
            # Validate input
            if not self.validate_input(state):
                return self.handle_error(
                    ValueError("Invalid financial research state"),
                    state
                )
            
            research_topic = state.get("research_topic", "")
            self.logger.info(f"Starting financial research for: {research_topic}")
            
            # Conduct research
            research_findings = self._conduct_financial_research(research_topic)
            
            # Validate findings
            if not research_findings:
                raise ValueError("No financial research findings generated")
            
            # Create completion command
            command = Command(
                goto="research_supervisor",
                update={
                    "financial_findings": research_findings,
                    "research_metadata": {
                        "completion_timestamp": datetime.now().isoformat(),
                        "agent": self.name,
                        "research_quality": self._assess_research_quality(research_findings)
                    }
                }
            )
            
            self.record_metrics(start_time, success=True)
            return command
            
        except Exception as e:
            self.record_metrics(start_time, success=False)
            return self.handle_error(e, state)
    
    def _conduct_financial_research(self, research_topic: str) -> Dict[str, Any]:
        """Conduct comprehensive financial research"""
        
        # Extract financial keywords
        financial_keywords = self._extract_financial_keywords(research_topic)
        
        # Search arXiv for relevant papers
        arxiv_results = self.arxiv_tool.search_papers(
            query=research_topic,
            category="q-fin",  # Quantitative Finance
            max_results=10
        )
        
        # Analyze papers with LLM
        paper_analysis = self._analyze_financial_papers(arxiv_results, research_topic)
        
        # Generate market analysis
        market_analysis = self._analyze_market_implications(research_topic, paper_analysis)
        
        # Risk assessment
        risk_assessment = self._conduct_risk_analysis(paper_analysis)
        
        # Economic indicators
        economic_indicators = self._extract_economic_indicators(paper_analysis)
        
        # Compile comprehensive findings
        findings = {
            "research_complete": True,
            "research_topic": research_topic,
            "key_findings": paper_analysis.get("key_findings", []),
            "market_analysis": market_analysis,
            "risk_assessment": risk_assessment,
            "economic_indicators": economic_indicators,
            "research_papers": [
                {
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []),
                    "abstract": paper.get("abstract", ""),
                    "url": paper.get("url", ""),
                    "relevance_score": paper.get("relevance_score", 0)
                }
                for paper in arxiv_results[:5]  # Top 5 papers
            ],
            "financial_specializations": self._identify_relevant_specializations(research_topic),
            "quality_indicators": {
                "paper_count": len(arxiv_results),
                "avg_relevance": sum(p.get("relevance_score", 0) for p in arxiv_results) / len(arxiv_results) if arxiv_results else 0,
                "market_factors": len(market_analysis.get("factors", [])),
                "risk_factors": len(risk_assessment.get("risks", []))
            },
            "sources": [paper.get("url", "") for paper in arxiv_results],
            "key_terms": financial_keywords,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return findings
    
    def _extract_financial_keywords(self, research_topic: str) -> List[str]:
        """Extract financial keywords from research topic"""
        
        prompt = f"""
        Extract financial and economic keywords from this research topic: {research_topic}
        
        Focus on:
        - Financial instruments and markets
        - Economic indicators and metrics
        - Investment strategies and techniques
        - Risk management concepts
        - Financial regulations and compliance
        
        Return only the keywords, one per line.
        """
        
        try:
            response = self.model.invoke(prompt)
            keywords = [line.strip() for line in response.content.split('\n') if line.strip()]
            return keywords[:20]  # Limit to top 20 keywords
        except Exception as e:
            self.logger.error(f"Financial keyword extraction failed: {e}")
            return []
    
    def _analyze_financial_papers(self, papers: List[Dict], research_topic: str) -> Dict[str, Any]:
        """Analyze financial research papers with LLM"""
        
        if not papers:
            return {"key_findings": [], "summary": "No papers found"}
        
        # Prepare paper summaries
        paper_summaries = []
        for paper in papers[:5]:  # Analyze top 5 papers
            summary = f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')[:500]}..."
            paper_summaries.append(summary)
        
        papers_text = "\n\n---\n\n".join(paper_summaries)
        
        prompt = f"""
        Analyze these financial research papers for the topic: {research_topic}
        
        Papers:
        {papers_text}
        
        Provide analysis in the following format:
        
        KEY FINDINGS:
        - [Finding 1]
        - [Finding 2]
        - [Finding 3]
        
        MARKET IMPLICATIONS:
        - [Implication 1]
        - [Implication 2]
        
        INVESTMENT INSIGHTS:
        - [Insight 1]
        - [Insight 2]
        
        SUMMARY:
        [2-3 sentence summary]
        """
        
        try:
            response = self.model.invoke(prompt)
            return self._parse_financial_analysis(response.content)
        except Exception as e:
            self.logger.error(f"Financial paper analysis failed: {e}")
            return {"key_findings": [], "summary": "Analysis failed"}
    
    def _parse_financial_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse LLM financial analysis response"""
        
        sections = {
            "key_findings": [],
            "market_implications": [],
            "investment_insights": [],
            "summary": ""
        }
        
        current_section = None
        
        for line in analysis_text.split('\n'):
            line = line.strip()
            
            if line.startswith('KEY FINDINGS:'):
                current_section = "key_findings"
            elif line.startswith('MARKET IMPLICATIONS:'):
                current_section = "market_implications"
            elif line.startswith('INVESTMENT INSIGHTS:'):
                current_section = "investment_insights"
            elif line.startswith('SUMMARY:'):
                current_section = "summary"
            elif line.startswith('- ') and current_section in ["key_findings", "market_implications", "investment_insights"]:
                sections[current_section].append(line[2:])
            elif current_section == "summary" and line:
                sections["summary"] += line + " "
        
        return sections
    
    def _analyze_market_implications(self, research_topic: str, paper_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market implications"""
        
        market_implications = paper_analysis.get("market_implications", [])
        key_findings = paper_analysis.get("key_findings", [])
        
        return {
            "factors": market_implications,
            "trends": self._identify_market_trends(market_implications),
            "opportunities": self._identify_opportunities(key_findings),
            "challenges": self._identify_challenges(key_findings),
            "sector_impact": self._assess_sector_impact(research_topic, market_implications)
        }
    
    def _identify_market_trends(self, implications: List[str]) -> List[str]:
        """Identify market trends from implications"""
        
        trends = []
        trend_keywords = ["trend", "growth", "decline", "increasing", "decreasing", "emerging"]
        
        for implication in implications:
            if any(keyword in implication.lower() for keyword in trend_keywords):
                trends.append(implication)
        
        return trends
    
    def _identify_opportunities(self, findings: List[str]) -> List[str]:
        """Identify investment opportunities"""
        
        opportunities = []
        opportunity_keywords = ["opportunity", "potential", "growth", "profit", "advantage"]
        
        for finding in findings:
            if any(keyword in finding.lower() for keyword in opportunity_keywords):
                opportunities.append(finding)
        
        return opportunities
    
    def _identify_challenges(self, findings: List[str]) -> List[str]:
        """Identify market challenges"""
        
        challenges = []
        challenge_keywords = ["risk", "challenge", "volatility", "uncertainty", "decline"]
        
        for finding in findings:
            if any(keyword in finding.lower() for keyword in challenge_keywords):
                challenges.append(finding)
        
        return challenges
    
    def _assess_sector_impact(self, research_topic: str, implications: List[str]) -> Dict[str, str]:
        """Assess impact on different sectors"""
        
        sectors = ["technology", "healthcare", "finance", "energy", "consumer", "industrial"]
        sector_impact = {}
        
        topic_lower = research_topic.lower()
        
        for sector in sectors:
            if sector in topic_lower:
                sector_impact[sector] = "high"
            elif any(sector in impl.lower() for impl in implications):
                sector_impact[sector] = "medium"
            else:
                sector_impact[sector] = "low"
        
        return sector_impact
    
    def _conduct_risk_analysis(self, paper_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive risk analysis"""
        
        key_findings = paper_analysis.get("key_findings", [])
        
        return {
            "risks": self._identify_risks(key_findings),
            "risk_levels": self._assess_risk_levels(key_findings),
            "mitigation_strategies": self._suggest_mitigation_strategies(key_findings),
            "regulatory_considerations": self._identify_regulatory_risks(key_findings)
        }
    
    def _identify_risks(self, findings: List[str]) -> List[str]:
        """Identify financial risks"""
        
        risks = []
        risk_keywords = ["risk", "volatility", "uncertainty", "loss", "default", "failure"]
        
        for finding in findings:
            if any(keyword in finding.lower() for keyword in risk_keywords):
                risks.append(finding)
        
        return risks
    
    def _assess_risk_levels(self, findings: List[str]) -> Dict[str, List[str]]:
        """Assess risk levels"""
        
        risk_levels = {"high": [], "medium": [], "low": []}
        
        high_risk_keywords = ["high risk", "extreme", "severe", "critical"]
        medium_risk_keywords = ["moderate", "medium", "significant"]
        
        for finding in findings:
            finding_lower = finding.lower()
            if any(keyword in finding_lower for keyword in high_risk_keywords):
                risk_levels["high"].append(finding)
            elif any(keyword in finding_lower for keyword in medium_risk_keywords):
                risk_levels["medium"].append(finding)
            else:
                risk_levels["low"].append(finding)
        
        return risk_levels
    
    def _suggest_mitigation_strategies(self, findings: List[str]) -> List[str]:
        """Suggest risk mitigation strategies"""
        
        strategies = []
        
        # Generic risk mitigation strategies
        if findings:
            strategies.extend([
                "Diversification across asset classes",
                "Regular portfolio rebalancing",
                "Risk monitoring and assessment",
                "Stop-loss mechanisms",
                "Hedging strategies"
            ])
        
        return strategies
    
    def _identify_regulatory_risks(self, findings: List[str]) -> List[str]:
        """Identify regulatory risks"""
        
        regulatory_risks = []
        regulatory_keywords = ["regulation", "compliance", "legal", "policy", "government"]
        
        for finding in findings:
            if any(keyword in finding.lower() for keyword in regulatory_keywords):
                regulatory_risks.append(finding)
        
        return regulatory_risks
    
    def _extract_economic_indicators(self, paper_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract economic indicators"""
        
        key_findings = paper_analysis.get("key_findings", [])
        
        return {
            "indicators": self._identify_economic_indicators(key_findings),
            "forecasts": self._extract_forecasts(key_findings),
            "correlations": self._identify_correlations(key_findings),
            "leading_indicators": self._identify_leading_indicators(key_findings)
        }
    
    def _identify_economic_indicators(self, findings: List[str]) -> List[str]:
        """Identify economic indicators"""
        
        indicators = []
        indicator_keywords = ["gdp", "inflation", "unemployment", "interest rate", "cpi", "ppi"]
        
        for finding in findings:
            if any(keyword in finding.lower() for keyword in indicator_keywords):
                indicators.append(finding)
        
        return indicators
    
    def _extract_forecasts(self, findings: List[str]) -> List[str]:
        """Extract economic forecasts"""
        
        forecasts = []
        forecast_keywords = ["forecast", "prediction", "projection", "outlook", "expect"]
        
        for finding in findings:
            if any(keyword in finding.lower() for keyword in forecast_keywords):
                forecasts.append(finding)
        
        return forecasts
    
    def _identify_correlations(self, findings: List[str]) -> List[str]:
        """Identify economic correlations"""
        
        correlations = []
        correlation_keywords = ["correlation", "relationship", "linked", "connected", "affects"]
        
        for finding in findings:
            if any(keyword in finding.lower() for keyword in correlation_keywords):
                correlations.append(finding)
        
        return correlations
    
    def _identify_leading_indicators(self, findings: List[str]) -> List[str]:
        """Identify leading economic indicators"""
        
        leading_indicators = []
        leading_keywords = ["leading", "predictive", "early", "signal", "precursor"]
        
        for finding in findings:
            if any(keyword in finding.lower() for keyword in leading_keywords):
                leading_indicators.append(finding)
        
        return leading_indicators
    
    def _identify_relevant_specializations(self, research_topic: str) -> List[str]:
        """Identify relevant financial specializations"""
        
        topic_lower = research_topic.lower()
        relevant_specializations = []
        
        specialization_keywords = {
            "market_analysis": ["market", "analysis", "trends"],
            "investment_strategies": ["investment", "strategy", "portfolio"],
            "risk_assessment": ["risk", "assessment", "management"],
            "economic_indicators": ["economic", "indicator", "forecast"],
            "financial_modeling": ["model", "modeling", "quantitative"],
            "algorithmic_trading": ["algorithm", "trading", "automated"]
        }
        
        for specialization, keywords in specialization_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                relevant_specializations.append(specialization)
        
        return relevant_specializations if relevant_specializations else ["general_financial_research"]
    
    def _assess_research_quality(self, findings: Dict[str, Any]) -> float:
        """Assess quality of financial research"""
        
        quality_score = 0.0
        
        # Paper quality (30%)
        paper_count = len(findings.get("research_papers", []))
        if paper_count >= 5:
            quality_score += 0.3
        elif paper_count >= 3:
            quality_score += 0.2
        elif paper_count >= 1:
            quality_score += 0.1
        
        # Key findings quality (25%)
        key_findings_count = len(findings.get("key_findings", []))
        if key_findings_count >= 5:
            quality_score += 0.25
        elif key_findings_count >= 3:
            quality_score += 0.15
        elif key_findings_count >= 1:
            quality_score += 0.1
        
        # Market analysis quality (25%)
        market_analysis = findings.get("market_analysis", {})
        if market_analysis.get("factors") and market_analysis.get("trends"):
            quality_score += 0.25
        elif market_analysis.get("factors") or market_analysis.get("trends"):
            quality_score += 0.15
        
        # Risk assessment quality (20%)
        risk_assessment = findings.get("risk_assessment", {})
        if risk_assessment.get("risks") and risk_assessment.get("mitigation_strategies"):
            quality_score += 0.20
        elif risk_assessment.get("risks") or risk_assessment.get("mitigation_strategies"):
            quality_score += 0.10
        
        return min(quality_score, 1.0)