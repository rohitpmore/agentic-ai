# Stage 3: Specialized Agent Implementation

**Timeline:** 5-6 hours  
**Status:** ⏳ Pending  
**Priority:** High

## 📋 Overview

This stage focuses on implementing the specialized agents that perform the core research and reporting tasks. We'll build the Medical/Pharmacy Researcher, Financial Researcher, Document Creator, and Summary Agent with their respective tools and integrations.

## 🎯 Key Deliverables

### ✅ Medical/Pharmacy Researcher Agent
### ✅ Financial Researcher Agent  
### ✅ Document Creator Agent
### ✅ Summary Agent
### ✅ arXiv API Integration Tool
### ✅ Document Generation Tools
### ✅ Unit Tests for All Agents

## 🔧 Implementation Details

### ✅ Medical/Pharmacy Researcher Agent
```python
# src/agents/research/medical_researcher.py
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from datetime import datetime
import logging

from src.agents.base_agent import BaseAgent
from src.tools.arxiv_tool import ArxivTool
from src.utils.error_handling import ErrorHandler
from config.settings import Settings

class MedicalResearcher(BaseAgent):
    """Medical and pharmaceutical researcher agent"""
    
    def __init__(self, settings: Settings):
        super().__init__("medical_researcher")
        self.settings = settings
        self.model = ChatOpenAI(
            model=settings.researcher_model,
            temperature=0.1,
            timeout=settings.timeout_seconds
        )
        self.arxiv_tool = ArxivTool()
        self.error_handler = ErrorHandler(max_retries=settings.max_retries)
        
        # Medical research specializations
        self.specializations = [
            "drug_interactions",
            "clinical_trials",
            "pharmaceutical_development",
            "medical_device_research",
            "biomarker_analysis",
            "therapeutic_interventions"
        ]
        
    def get_required_fields(self) -> List[str]:
        return ["research_topic", "research_status"]
        
    def process(self, state: Dict[str, Any]) -> Command:
        """Process medical research request"""
        
        start_time = datetime.now()
        
        try:
            # Validate input
            if not self.validate_input(state):
                return self.handle_error(
                    ValueError("Invalid medical research state"),
                    state
                )
            
            research_topic = state.get("research_topic", "")
            self.logger.info(f"Starting medical research for: {research_topic}")
            
            # Conduct research
            research_findings = self._conduct_medical_research(research_topic)
            
            # Validate findings
            if not research_findings:
                raise ValueError("No medical research findings generated")
            
            # Create completion command
            command = Command(
                goto="research_supervisor",
                update={
                    "medical_findings": research_findings,
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
    
    def _conduct_medical_research(self, research_topic: str) -> Dict[str, Any]:
        """Conduct comprehensive medical research"""
        
        # Extract medical keywords
        medical_keywords = self._extract_medical_keywords(research_topic)
        
        # Search arXiv for relevant papers
        arxiv_results = self.arxiv_tool.search_papers(
            query=research_topic,
            category="q-bio",  # Quantitative Biology
            max_results=10
        )
        
        # Analyze papers with LLM
        paper_analysis = self._analyze_medical_papers(arxiv_results, research_topic)
        
        # Generate drug interaction analysis
        drug_interactions = self._analyze_drug_interactions(research_topic, paper_analysis)
        
        # Clinical trial insights
        clinical_insights = self._extract_clinical_insights(paper_analysis)
        
        # Compile comprehensive findings
        findings = {
            "research_complete": True,
            "research_topic": research_topic,
            "key_findings": paper_analysis.get("key_findings", []),
            "drug_interactions": drug_interactions,
            "clinical_insights": clinical_insights,
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
            "medical_specializations": self._identify_relevant_specializations(research_topic),
            "quality_indicators": {
                "paper_count": len(arxiv_results),
                "avg_relevance": sum(p.get("relevance_score", 0) for p in arxiv_results) / len(arxiv_results) if arxiv_results else 0,
                "clinical_trial_count": len(clinical_insights.get("trials", [])),
                "drug_interaction_count": len(drug_interactions.get("interactions", []))
            },
            "sources": [paper.get("url", "") for paper in arxiv_results],
            "key_terms": medical_keywords,
            "research_timestamp": datetime.now().isoformat()
        }
        
        return findings
    
    def _extract_medical_keywords(self, research_topic: str) -> List[str]:
        """Extract medical keywords from research topic"""
        
        prompt = f"""
        Extract medical and pharmaceutical keywords from this research topic: {research_topic}
        
        Focus on:
        - Medical conditions and diseases
        - Drug names and pharmaceutical compounds
        - Medical procedures and treatments
        - Anatomical terms
        - Clinical terminology
        
        Return only the keywords, one per line.
        """
        
        try:
            response = self.model.invoke(prompt)
            keywords = [line.strip() for line in response.content.split('\n') if line.strip()]
            return keywords[:20]  # Limit to top 20 keywords
        except Exception as e:
            self.logger.error(f"Keyword extraction failed: {e}")
            return []
    
    def _analyze_medical_papers(self, papers: List[Dict], research_topic: str) -> Dict[str, Any]:
        """Analyze medical research papers with LLM"""
        
        if not papers:
            return {"key_findings": [], "summary": "No papers found"}
        
        # Prepare paper summaries
        paper_summaries = []
        for paper in papers[:5]:  # Analyze top 5 papers
            summary = f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')[:500]}..."
            paper_summaries.append(summary)
        
        papers_text = "\n\n---\n\n".join(paper_summaries)
        
        prompt = f"""
        Analyze these medical research papers for the topic: {research_topic}
        
        Papers:
        {papers_text}
        
        Provide analysis in the following format:
        
        KEY FINDINGS:
        - [Finding 1]
        - [Finding 2]
        - [Finding 3]
        
        CLINICAL IMPLICATIONS:
        - [Implication 1]
        - [Implication 2]
        
        RESEARCH GAPS:
        - [Gap 1]
        - [Gap 2]
        
        SUMMARY:
        [2-3 sentence summary]
        """
        
        try:
            response = self.model.invoke(prompt)
            return self._parse_medical_analysis(response.content)
        except Exception as e:
            self.logger.error(f"Paper analysis failed: {e}")
            return {"key_findings": [], "summary": "Analysis failed"}
    
    def _parse_medical_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse LLM analysis response"""
        
        sections = {
            "key_findings": [],
            "clinical_implications": [],
            "research_gaps": [],
            "summary": ""
        }
        
        current_section = None
        
        for line in analysis_text.split('\n'):
            line = line.strip()
            
            if line.startswith('KEY FINDINGS:'):
                current_section = "key_findings"
            elif line.startswith('CLINICAL IMPLICATIONS:'):
                current_section = "clinical_implications"
            elif line.startswith('RESEARCH GAPS:'):
                current_section = "research_gaps"
            elif line.startswith('SUMMARY:'):
                current_section = "summary"
            elif line.startswith('- ') and current_section in ["key_findings", "clinical_implications", "research_gaps"]:
                sections[current_section].append(line[2:])
            elif current_section == "summary" and line:
                sections["summary"] += line + " "
        
        return sections
    
    def _analyze_drug_interactions(self, research_topic: str, paper_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential drug interactions"""
        
        key_findings = paper_analysis.get("key_findings", [])
        
        prompt = f"""
        Based on this medical research topic: {research_topic}
        And these key findings: {'; '.join(key_findings)}
        
        Identify potential drug interactions and contraindications.
        
        Format your response as:
        
        DRUG INTERACTIONS:
        - [Drug A] + [Drug B]: [Interaction description]
        
        CONTRAINDICATIONS:
        - [Condition/Drug]: [Contraindication description]
        
        SAFETY CONSIDERATIONS:
        - [Safety point 1]
        - [Safety point 2]
        """
        
        try:
            response = self.model.invoke(prompt)
            return self._parse_drug_interactions(response.content)
        except Exception as e:
            self.logger.error(f"Drug interaction analysis failed: {e}")
            return {"interactions": [], "contraindications": [], "safety_considerations": []}
    
    def _parse_drug_interactions(self, interaction_text: str) -> Dict[str, Any]:
        """Parse drug interaction analysis"""
        
        result = {
            "interactions": [],
            "contraindications": [],
            "safety_considerations": []
        }
        
        current_section = None
        
        for line in interaction_text.split('\n'):
            line = line.strip()
            
            if line.startswith('DRUG INTERACTIONS:'):
                current_section = "interactions"
            elif line.startswith('CONTRAINDICATIONS:'):
                current_section = "contraindications"
            elif line.startswith('SAFETY CONSIDERATIONS:'):
                current_section = "safety_considerations"
            elif line.startswith('- ') and current_section:
                result[current_section].append(line[2:])
        
        return result
    
    def _extract_clinical_insights(self, paper_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract clinical trial and research insights"""
        
        clinical_implications = paper_analysis.get("clinical_implications", [])
        
        return {
            "trials": [
                {
                    "type": "clinical_trial",
                    "description": impl,
                    "relevance": "high" if any(keyword in impl.lower() for keyword in ["trial", "study", "patients"]) else "medium"
                }
                for impl in clinical_implications
            ],
            "patient_populations": self._identify_patient_populations(clinical_implications),
            "treatment_protocols": self._extract_treatment_protocols(clinical_implications),
            "efficacy_measures": self._identify_efficacy_measures(clinical_implications)
        }
    
    def _identify_patient_populations(self, clinical_implications: List[str]) -> List[str]:
        """Identify relevant patient populations"""
        
        populations = []
        population_keywords = ["adults", "children", "elderly", "patients with", "individuals with"]
        
        for implication in clinical_implications:
            for keyword in population_keywords:
                if keyword in implication.lower():
                    populations.append(implication)
                    break
        
        return populations
    
    def _extract_treatment_protocols(self, clinical_implications: List[str]) -> List[str]:
        """Extract treatment protocols"""
        
        protocols = []
        protocol_keywords = ["treatment", "therapy", "protocol", "administration", "dosage"]
        
        for implication in clinical_implications:
            if any(keyword in implication.lower() for keyword in protocol_keywords):
                protocols.append(implication)
        
        return protocols
    
    def _identify_efficacy_measures(self, clinical_implications: List[str]) -> List[str]:
        """Identify efficacy measures"""
        
        measures = []
        efficacy_keywords = ["efficacy", "effectiveness", "outcome", "response", "improvement"]
        
        for implication in clinical_implications:
            if any(keyword in implication.lower() for keyword in efficacy_keywords):
                measures.append(implication)
        
        return measures
    
    def _identify_relevant_specializations(self, research_topic: str) -> List[str]:
        """Identify relevant medical specializations"""
        
        topic_lower = research_topic.lower()
        relevant_specializations = []
        
        specialization_keywords = {
            "drug_interactions": ["drug", "medication", "pharmaceutical", "interaction"],
            "clinical_trials": ["clinical", "trial", "study", "patient"],
            "pharmaceutical_development": ["development", "discovery", "synthesis"],
            "medical_device_research": ["device", "equipment", "technology"],
            "biomarker_analysis": ["biomarker", "diagnostic", "screening"],
            "therapeutic_interventions": ["therapy", "treatment", "intervention"]
        }
        
        for specialization, keywords in specialization_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                relevant_specializations.append(specialization)
        
        return relevant_specializations if relevant_specializations else ["general_medical_research"]
    
    def _assess_research_quality(self, findings: Dict[str, Any]) -> float:
        """Assess quality of medical research"""
        
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
        
        # Clinical insights quality (25%)
        clinical_insights = findings.get("clinical_insights", {})
        if clinical_insights.get("trials") and clinical_insights.get("patient_populations"):
            quality_score += 0.25
        elif clinical_insights.get("trials") or clinical_insights.get("patient_populations"):
            quality_score += 0.15
        
        # Drug interaction analysis quality (20%)
        drug_interactions = findings.get("drug_interactions", {})
        if drug_interactions.get("interactions") and drug_interactions.get("contraindications"):
            quality_score += 0.20
        elif drug_interactions.get("interactions") or drug_interactions.get("contraindications"):
            quality_score += 0.10
        
        return min(quality_score, 1.0)
```

### ✅ Financial Researcher Agent
```python
# src/agents/research/financial_researcher.py
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from datetime import datetime
import logging

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
    
    def _assess_risk_levels(self, findings: List[str]) -> Dict[str, str]:
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
```

### ✅ Document Creator Agent
```python
# src/agents/reporting/document_creator.py
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from datetime import datetime
import os
import logging

from src.agents.base_agent import BaseAgent
from src.tools.document_tools import DocumentTool
from src.utils.error_handling import ErrorHandler
from config.settings import Settings

class DocumentCreator(BaseAgent):
    """Document creator agent for generating professional reports"""
    
    def __init__(self, settings: Settings):
        super().__init__("document_creator")
        self.settings = settings
        self.model = ChatOpenAI(
            model=settings.reporter_model,
            temperature=0.1,
            timeout=settings.timeout_seconds
        )
        self.document_tool = DocumentTool(settings)
        self.error_handler = ErrorHandler(max_retries=settings.max_retries)
        
    def get_required_fields(self) -> List[str]:
        return ["research_data", "report_status"]
        
    def process(self, state: Dict[str, Any]) -> Command:
        """Process document creation request"""
        
        start_time = datetime.now()
        
        try:
            # Validate input
            if not self.validate_input(state):
                return self.handle_error(
                    ValueError("Invalid document creation state"),
                    state
                )
            
            research_data = state.get("research_data", {})
            self.logger.info("Starting document creation")
            
            # Generate document
            document_path = self._create_professional_document(research_data)
            
            if not document_path:
                raise ValueError("Document creation failed")
            
            # Create completion command
            command = Command(
                goto="reporting_supervisor",
                update={
                    "document_path": document_path,
                    "report_metadata": {
                        "completion_timestamp": datetime.now().isoformat(),
                        "agent": self.name,
                        "document_type": "comprehensive_report",
                        "file_size": os.path.getsize(document_path) if os.path.exists(document_path) else 0
                    }
                }
            )
            
            self.record_metrics(start_time, success=True)
            return command
            
        except Exception as e:
            self.record_metrics(start_time, success=False)
            return self.handle_error(e, state)
    
    def _create_professional_document(self, research_data: Dict[str, Any]) -> str:
        """Create professional document from research data"""
        
        # Extract research findings
        medical_findings = research_data.get("medical_findings", {})
        financial_findings = research_data.get("financial_findings", {})
        
        # Generate document structure
        document_structure = self._generate_document_structure(medical_findings, financial_findings)
        
        # Create document using document tool
        document_path = self.document_tool.create_document(
            structure=document_structure,
            format="pdf",
            template="professional_report"
        )
        
        return document_path
    
    def _generate_document_structure(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive document structure"""
        
        return {
            "title": self._generate_document_title(medical_findings, financial_findings),
            "executive_summary": self._generate_executive_summary(medical_findings, financial_findings),
            "sections": [
                {
                    "title": "Medical Research Findings",
                    "content": self._format_medical_section(medical_findings)
                },
                {
                    "title": "Financial Research Findings",
                    "content": self._format_financial_section(financial_findings)
                },
                {
                    "title": "Cross-Domain Analysis",
                    "content": self._generate_cross_domain_analysis(medical_findings, financial_findings)
                },
                {
                    "title": "Recommendations",
                    "content": self._generate_recommendations(medical_findings, financial_findings)
                }
            ],
            "appendices": self._generate_appendices(medical_findings, financial_findings),
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "version": "1.0",
                "authors": ["Medical Research Agent", "Financial Research Agent"],
                "document_type": "Research Report"
            }
        }
    
    def _generate_document_title(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> str:
        """Generate appropriate document title"""
        
        medical_topic = medical_findings.get("research_topic", "")
        financial_topic = financial_findings.get("research_topic", "")
        
        if medical_topic and financial_topic:
            return f"Comprehensive Analysis: {medical_topic} and {financial_topic}"
        elif medical_topic:
            return f"Medical Research Report: {medical_topic}"
        elif financial_topic:
            return f"Financial Research Report: {financial_topic}"
        else:
            return "Multi-Domain Research Report"
    
    def _generate_executive_summary(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> str:
        """Generate executive summary"""
        
        summary_parts = []
        
        # Medical summary
        if medical_findings.get("key_findings"):
            medical_summary = f"Medical research identified {len(medical_findings['key_findings'])} key findings"
            summary_parts.append(medical_summary)
        
        # Financial summary
        if financial_findings.get("key_findings"):
            financial_summary = f"Financial analysis revealed {len(financial_findings['key_findings'])} key insights"
            summary_parts.append(financial_summary)
        
        # Quality indicators
        total_papers = len(medical_findings.get("research_papers", [])) + len(financial_findings.get("research_papers", []))
        if total_papers > 0:
            summary_parts.append(f"Analysis based on {total_papers} research papers")
        
        return ". ".join(summary_parts) + "."
    
    def _format_medical_section(self, medical_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Format medical research section"""
        
        return {
            "key_findings": medical_findings.get("key_findings", []),
            "clinical_insights": medical_findings.get("clinical_insights", {}),
            "drug_interactions": medical_findings.get("drug_interactions", {}),
            "quality_score": medical_findings.get("quality_indicators", {}).get("avg_relevance", 0),
            "source_count": len(medical_findings.get("research_papers", [])),
            "specializations": medical_findings.get("medical_specializations", [])
        }
    
    def _format_financial_section(self, financial_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Format financial research section"""
        
        return {
            "key_findings": financial_findings.get("key_findings", []),
            "market_analysis": financial_findings.get("market_analysis", {}),
            "risk_assessment": financial_findings.get("risk_assessment", {}),
            "economic_indicators": financial_findings.get("economic_indicators", {}),
            "quality_score": financial_findings.get("quality_indicators", {}).get("avg_relevance", 0),
            "source_count": len(financial_findings.get("research_papers", [])),
            "specializations": financial_findings.get("financial_specializations", [])
        }
    
    def _generate_cross_domain_analysis(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cross-domain analysis"""
        
        return {
            "synergies": self._identify_synergies(medical_findings, financial_findings),
            "conflicts": self._identify_conflicts(medical_findings, financial_findings),
            "investment_implications": self._analyze_investment_implications(medical_findings, financial_findings),
            "regulatory_considerations": self._analyze_regulatory_overlap(medical_findings, financial_findings)
        }
    
    def _identify_synergies(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> List[str]:
        """Identify synergies between medical and financial findings"""
        
        synergies = []
        
        # Look for common themes
        medical_terms = set(medical_findings.get("key_terms", []))
        financial_terms = set(financial_findings.get("key_terms", []))
        
        common_terms = medical_terms.intersection(financial_terms)
        if common_terms:
            synergies.append(f"Common research themes: {', '.join(common_terms)}")
        
        # Healthcare investment opportunities
        if medical_findings.get("clinical_insights") and financial_findings.get("market_analysis"):
            synergies.append("Healthcare investment opportunities identified from clinical research")
        
        return synergies
    
    def _identify_conflicts(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> List[str]:
        """Identify conflicts between domains"""
        
        conflicts = []
        
        # Risk vs. benefit analysis
        medical_risks = medical_findings.get("drug_interactions", {}).get("safety_considerations", [])
        financial_risks = financial_findings.get("risk_assessment", {}).get("risks", [])
        
        if medical_risks and financial_risks:
            conflicts.append("Potential conflicts between medical safety and financial risk considerations")
        
        return conflicts
    
    def _analyze_investment_implications(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> List[str]:
        """Analyze investment implications"""
        
        implications = []
        
        # Medical breakthroughs with financial impact
        if medical_findings.get("key_findings"):
            implications.append("Medical research findings may have investment implications")
        
        # Market opportunities in healthcare
        if financial_findings.get("market_analysis", {}).get("opportunities"):
            implications.append("Financial analysis identifies healthcare market opportunities")
        
        return implications
    
    def _analyze_regulatory_overlap(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> List[str]:
        """Analyze regulatory considerations"""
        
        regulatory_items = []
        
        # Medical regulations
        if medical_findings.get("clinical_insights"):
            regulatory_items.append("Medical regulatory compliance required")
        
        # Financial regulations
        if financial_findings.get("risk_assessment", {}).get("regulatory_considerations"):
            regulatory_items.append("Financial regulatory considerations identified")
        
        return regulatory_items
    
    def _generate_recommendations(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Based on medical findings
        if medical_findings.get("clinical_insights"):
            recommendations.append("Consider clinical trial implications for investment decisions")
        
        # Based on financial findings
        if financial_findings.get("market_analysis", {}).get("opportunities"):
            recommendations.append("Explore identified market opportunities")
        
        # Risk mitigation
        medical_risks = medical_findings.get("drug_interactions", {}).get("safety_considerations", [])
        financial_risks = financial_findings.get("risk_assessment", {}).get("risks", [])
        
        if medical_risks or financial_risks:
            recommendations.append("Implement comprehensive risk management strategies")
        
        # Further research
        recommendations.append("Conduct additional research in identified areas of interest")
        
        return recommendations
    
    def _generate_appendices(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate document appendices"""
        
        appendices = []
        
        # Medical sources
        if medical_findings.get("research_papers"):
            appendices.append({
                "title": "Medical Research Sources",
                "content": medical_findings["research_papers"]
            })
        
        # Financial sources
        if financial_findings.get("research_papers"):
            appendices.append({
                "title": "Financial Research Sources",
                "content": financial_findings["research_papers"]
            })
        
        # Methodology
        appendices.append({
            "title": "Research Methodology",
            "content": {
                "medical_approach": "Literature review using arXiv database",
                "financial_approach": "Quantitative analysis of financial research papers",
                "analysis_framework": "Multi-agent collaborative research system"
            }
        })
        
        return appendices
```

### ✅ Summary Agent
```python
# src/agents/reporting/summarizer.py
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from datetime import datetime
import logging

from src.agents.base_agent import BaseAgent
from src.utils.error_handling import ErrorHandler
from config.settings import Settings

class Summarizer(BaseAgent):
    """Summary agent for creating concise summaries of research findings"""
    
    def __init__(self, settings: Settings):
        super().__init__("summarizer")
        self.settings = settings
        self.model = ChatOpenAI(
            model=settings.reporter_model,
            temperature=0.1,
            timeout=settings.timeout_seconds
        )
        self.error_handler = ErrorHandler(max_retries=settings.max_retries)
        
    def get_required_fields(self) -> List[str]:
        return ["research_data", "report_status"]
        
    def process(self, state: Dict[str, Any]) -> Command:
        """Process summary creation request"""
        
        start_time = datetime.now()
        
        try:
            # Validate input
            if not self.validate_input(state):
                return self.handle_error(
                    ValueError("Invalid summary creation state"),
                    state
                )
            
            research_data = state.get("research_data", {})
            self.logger.info("Starting summary creation")
            
            # Generate summary
            summary = self._create_comprehensive_summary(research_data)
            
            if not summary:
                raise ValueError("Summary creation failed")
            
            # Create completion command
            command = Command(
                goto="reporting_supervisor",
                update={
                    "summary": summary,
                    "report_metadata": {
                        "completion_timestamp": datetime.now().isoformat(),
                        "agent": self.name,
                        "summary_length": len(summary),
                        "summary_type": "executive_summary"
                    }
                }
            )
            
            self.record_metrics(start_time, success=True)
            return command
            
        except Exception as e:
            self.record_metrics(start_time, success=False)
            return self.handle_error(e, state)
    
    def _create_comprehensive_summary(self, research_data: Dict[str, Any]) -> str:
        """Create comprehensive summary of research findings"""
        
        medical_findings = research_data.get("medical_findings", {})
        financial_findings = research_data.get("financial_findings", {})
        
        # Generate different types of summaries
        executive_summary = self._generate_executive_summary(medical_findings, financial_findings)
        key_points = self._extract_key_points(medical_findings, financial_findings)
        recommendations = self._generate_summary_recommendations(medical_findings, financial_findings)
        
        # Combine into comprehensive summary
        summary_parts = [
            "# Executive Summary",
            executive_summary,
            "",
            "## Key Findings",
            *[f"• {point}" for point in key_points],
            "",
            "## Recommendations",
            *[f"• {rec}" for rec in recommendations],
            "",
            "## Research Quality",
            self._generate_quality_assessment(medical_findings, financial_findings)
        ]
        
        return "\n".join(summary_parts)
    
    def _generate_executive_summary(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> str:
        """Generate executive summary using LLM"""
        
        # Prepare research context
        context = self._prepare_research_context(medical_findings, financial_findings)
        
        prompt = f"""
        Create a concise executive summary (2-3 paragraphs) based on the following research findings:
        
        {context}
        
        The summary should:
        1. Highlight the most important findings from both domains
        2. Identify key insights and implications
        3. Mention any cross-domain connections
        4. Be accessible to executive-level audience
        
        Write in a professional, clear, and engaging style.
        """
        
        try:
            response = self.model.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            self.logger.error(f"Executive summary generation failed: {e}")
            return self._generate_fallback_summary(medical_findings, financial_findings)
    
    def _prepare_research_context(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> str:
        """Prepare research context for LLM"""
        
        context_parts = []
        
        # Medical context
        if medical_findings.get("key_findings"):
            context_parts.append("MEDICAL RESEARCH FINDINGS:")
            context_parts.extend([f"- {finding}" for finding in medical_findings["key_findings"][:5]])
            context_parts.append("")
        
        # Financial context
        if financial_findings.get("key_findings"):
            context_parts.append("FINANCIAL RESEARCH FINDINGS:")
            context_parts.extend([f"- {finding}" for finding in financial_findings["key_findings"][:5]])
            context_parts.append("")
        
        # Research quality
        medical_quality = medical_findings.get("quality_indicators", {})
        financial_quality = financial_findings.get("quality_indicators", {})
        
        if medical_quality or financial_quality:
            context_parts.append("RESEARCH QUALITY:")
            if medical_quality.get("paper_count"):
                context_parts.append(f"- Medical papers analyzed: {medical_quality['paper_count']}")
            if financial_quality.get("paper_count"):
                context_parts.append(f"- Financial papers analyzed: {financial_quality['paper_count']}")
        
        return "\n".join(context_parts)
    
    def _extract_key_points(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> List[str]:
        """Extract key points from research findings"""
        
        key_points = []
        
        # Medical key points
        medical_key_findings = medical_findings.get("key_findings", [])
        if medical_key_findings:
            key_points.append(f"Medical research identified {len(medical_key_findings)} key findings")
            key_points.extend(medical_key_findings[:3])  # Top 3 medical findings
        
        # Financial key points
        financial_key_findings = financial_findings.get("key_findings", [])
        if financial_key_findings:
            key_points.append(f"Financial analysis revealed {len(financial_key_findings)} key insights")
            key_points.extend(financial_key_findings[:3])  # Top 3 financial findings
        
        # Clinical insights
        clinical_insights = medical_findings.get("clinical_insights", {})
        if clinical_insights.get("trials"):
            key_points.append(f"Clinical trial data available for {len(clinical_insights['trials'])} studies")
        
        # Market implications
        market_analysis = financial_findings.get("market_analysis", {})
        if market_analysis.get("opportunities"):
            key_points.append(f"Market opportunities identified in {len(market_analysis['opportunities'])} areas")
        
        return key_points[:10]  # Limit to top 10 key points
    
    def _generate_summary_recommendations(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> List[str]:
        """Generate summary recommendations"""
        
        recommendations = []
        
        # Medical recommendations
        if medical_findings.get("drug_interactions", {}).get("safety_considerations"):
            recommendations.append("Consider drug interaction and safety implications")
        
        # Financial recommendations
        if financial_findings.get("risk_assessment", {}).get("risks"):
            recommendations.append("Implement risk management strategies")
        
        # Cross-domain recommendations
        if medical_findings.get("clinical_insights") and financial_findings.get("market_analysis"):
            recommendations.append("Explore investment opportunities in healthcare innovation")
        
        # Further research
        recommendations.append("Conduct additional research in identified priority areas")
        
        return recommendations
    
    def _generate_quality_assessment(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> str:
        """Generate quality assessment"""
        
        quality_parts = []
        
        # Medical quality
        medical_quality = medical_findings.get("quality_indicators", {})
        if medical_quality:
            medical_score = medical_quality.get("avg_relevance", 0)
            paper_count = medical_quality.get("paper_count", 0)
            quality_parts.append(f"Medical research quality: {medical_score:.2f}/1.0 ({paper_count} papers)")
        
        # Financial quality
        financial_quality = financial_findings.get("quality_indicators", {})
        if financial_quality:
            financial_score = financial_quality.get("avg_relevance", 0)
            paper_count = financial_quality.get("paper_count", 0)
            quality_parts.append(f"Financial research quality: {financial_score:.2f}/1.0 ({paper_count} papers)")
        
        # Overall assessment
        if medical_quality and financial_quality:
            overall_score = (medical_quality.get("avg_relevance", 0) + financial_quality.get("avg_relevance", 0)) / 2
            quality_parts.append(f"Overall research quality: {overall_score:.2f}/1.0")
        
        return " | ".join(quality_parts) if quality_parts else "Quality assessment not available"
    
    def _generate_fallback_summary(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> str:
        """Generate fallback summary when LLM fails"""
        
        summary_parts = []
        
        # Basic summary
        summary_parts.append("Research Summary")
        summary_parts.append("=" * 16)
        
        # Medical findings
        if medical_findings.get("key_findings"):
            summary_parts.append(f"Medical research identified {len(medical_findings['key_findings'])} key findings.")
        
        # Financial findings
        if financial_findings.get("key_findings"):
            summary_parts.append(f"Financial analysis revealed {len(financial_findings['key_findings'])} key insights.")
        
        # Research scope
        total_papers = len(medical_findings.get("research_papers", [])) + len(financial_findings.get("research_papers", []))
        if total_papers > 0:
            summary_parts.append(f"Analysis based on {total_papers} research papers from academic sources.")
        
        return "\n".join(summary_parts)
```

## 🧪 Testing Implementation

### ✅ Unit Tests for Specialized Agents
```python
# tests/unit/test_specialized_agents.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.research.medical_researcher import MedicalResearcher
from src.agents.research.financial_researcher import FinancialResearcher
from src.agents.reporting.document_creator import DocumentCreator
from src.agents.reporting.summarizer import Summarizer
from config.settings import Settings

class TestMedicalResearcher:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test-key",
            researcher_model="gpt-4",
            max_retries=3
        )
    
    @pytest.fixture
    def researcher(self, settings):
        return MedicalResearcher(settings)
    
    @patch('src.tools.arxiv_tool.ArxivTool.search_papers')
    @patch('langchain_openai.ChatOpenAI.invoke')
    def test_medical_research_process(self, mock_invoke, mock_search, researcher):
        # Mock arXiv search results
        mock_search.return_value = [
            {
                "title": "Medical AI Research",
                "authors": ["Author 1"],
                "abstract": "Abstract content",
                "url": "http://example.com",
                "relevance_score": 0.8
            }
        ]
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """
        KEY FINDINGS:
        - Finding 1
        - Finding 2
        
        CLINICAL IMPLICATIONS:
        - Implication 1
        
        RESEARCH GAPS:
        - Gap 1
        
        SUMMARY:
        Test summary
        """
        mock_invoke.return_value = mock_response
        
        state = {
            "research_topic": "AI in medical diagnostics",
            "research_status": "pending"
        }
        
        result = researcher.process(state)
        
        assert result.goto == "research_supervisor"
        assert "medical_findings" in result.update
        assert result.update["medical_findings"]["research_complete"] is True
        
        mock_search.assert_called_once()
        mock_invoke.assert_called()

class TestFinancialResearcher:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test-key",
            researcher_model="gpt-4",
            max_retries=3
        )
    
    @pytest.fixture
    def researcher(self, settings):
        return FinancialResearcher(settings)
    
    @patch('src.tools.arxiv_tool.ArxivTool.search_papers')
    @patch('langchain_openai.ChatOpenAI.invoke')
    def test_financial_research_process(self, mock_invoke, mock_search, researcher):
        # Mock arXiv search results
        mock_search.return_value = [
            {
                "title": "Financial AI Research",
                "authors": ["Author 1"],
                "abstract": "Financial abstract",
                "url": "http://example.com",
                "relevance_score": 0.9
            }
        ]
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """
        KEY FINDINGS:
        - Market finding 1
        - Market finding 2
        
        MARKET IMPLICATIONS:
        - Implication 1
        
        INVESTMENT INSIGHTS:
        - Insight 1
        
        SUMMARY:
        Financial summary
        """
        mock_invoke.return_value = mock_response
        
        state = {
            "research_topic": "AI in financial markets",
            "research_status": "pending"
        }
        
        result = researcher.process(state)
        
        assert result.goto == "research_supervisor"
        assert "financial_findings" in result.update
        assert result.update["financial_findings"]["research_complete"] is True

class TestDocumentCreator:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test-key",
            reporter_model="gpt-3.5-turbo",
            output_directory="./test_outputs"
        )
    
    @pytest.fixture
    def creator(self, settings):
        return DocumentCreator(settings)
    
    @patch('src.tools.document_tools.DocumentTool.create_document')
    def test_document_creation(self, mock_create_doc, creator):
        # Mock document creation
        mock_create_doc.return_value = "/path/to/document.pdf"
        
        state = {
            "research_data": {
                "medical_findings": {"key_findings": ["Finding 1"]},
                "financial_findings": {"key_findings": ["Finding 2"]}
            },
            "report_status": "pending"
        }
        
        result = creator.process(state)
        
        assert result.goto == "reporting_supervisor"
        assert "document_path" in result.update
        assert result.update["document_path"] == "/path/to/document.pdf"
        
        mock_create_doc.assert_called_once()

class TestSummarizer:
    @pytest.fixture
    def settings(self):
        return Settings(
            openai_api_key="test-key",
            reporter_model="gpt-3.5-turbo"
        )
    
    @pytest.fixture
    def summarizer(self, settings):
        return Summarizer(settings)
    
    @patch('langchain_openai.ChatOpenAI.invoke')
    def test_summary_creation(self, mock_invoke, summarizer):
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "This is a comprehensive summary of the research findings."
        mock_invoke.return_value = mock_response
        
        state = {
            "research_data": {
                "medical_findings": {"key_findings": ["Medical finding 1", "Medical finding 2"]},
                "financial_findings": {"key_findings": ["Financial finding 1", "Financial finding 2"]}
            },
            "report_status": "pending"
        }
        
        result = summarizer.process(state)
        
        assert result.goto == "reporting_supervisor"
        assert "summary" in result.update
        assert len(result.update["summary"]) > 0
        
        mock_invoke.assert_called_once()
```

## 🎯 Success Criteria

### Functional Requirements:
- [ ] Medical Researcher conducts comprehensive medical literature review
- [ ] Financial Researcher performs thorough financial analysis
- [ ] Document Creator generates professional PDF/DOCX documents
- [ ] Summary Agent creates concise, executive-level summaries
- [ ] All agents integrate with arXiv API successfully
- [ ] Document generation works with multiple formats

### Quality Requirements:
- [ ] Unit tests achieve >90% coverage for all agents
- [ ] Research quality scoring functions properly
- [ ] Error handling prevents agent failures
- [ ] LLM integration handles API failures gracefully
- [ ] Generated documents are properly formatted

### Performance Requirements:
- [ ] Medical research completes in <3 minutes
- [ ] Financial research completes in <3 minutes
- [ ] Document generation completes in <2 minutes
- [ ] Summary creation completes in <1 minute
- [ ] API calls have proper rate limiting

## 📊 Stage 3 Metrics

### Time Allocation:
- Medical Researcher implementation: 90 minutes
- Financial Researcher implementation: 90 minutes
- Document Creator implementation: 75 minutes
- Summary Agent implementation: 60 minutes
- arXiv Tool integration: 45 minutes
- Document Tools implementation: 45 minutes
- Unit tests: 90 minutes

### Success Indicators:
- All agents process requests without errors
- Research findings are comprehensive and relevant
- Documents are professionally formatted
- Summaries are concise and informative
- API integrations work reliably
- Performance metrics meet targets

## 🔄 Next Steps

After completing Stage 3, proceed to:
1. **Stage 4**: LangGraph Integration & Workflow Assembly
2. Build hierarchical team subgraphs
3. Implement state transition logic
4. Add parallel execution capabilities

---

*This stage implements the core intelligence of the multi-agent system, providing sophisticated research and reporting capabilities that form the foundation of the workflow.*