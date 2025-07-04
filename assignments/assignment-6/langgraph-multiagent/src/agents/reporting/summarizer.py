from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from datetime import datetime

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