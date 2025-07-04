from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from datetime import datetime
import os

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
        """Generate compelling document title using LLM"""
        
        context = self._prepare_title_context(medical_findings, financial_findings)
        
        prompt = f"""
Create a compelling, professional document title based on this research:

{context}

Requirements:
- Executive-level audience
- Professional and engaging
- Reflects both domains if present
- Maximum 15 words
- Captures key insights

Return only the title, no explanation.
"""
        
        try:
            response = self.model.invoke(prompt)
            title = response.content.strip().strip('"')
            return title if title else self._generate_fallback_title(medical_findings, financial_findings)
        except Exception as e:
            self.logger.error(f"LLM title generation failed: {e}")
            return self._generate_fallback_title(medical_findings, financial_findings)
    
    def _prepare_title_context(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> str:
        """Prepare context for title generation"""
        
        context_parts = []
        
        medical_topic = medical_findings.get("research_topic", "")
        medical_key_findings = medical_findings.get("key_findings", [])[:3]
        if medical_topic:
            context_parts.append(f"Medical Topic: {medical_topic}")
        if medical_key_findings:
            context_parts.append(f"Medical Insights: {'; '.join(medical_key_findings)}")
        
        financial_topic = financial_findings.get("research_topic", "")
        financial_key_findings = financial_findings.get("key_findings", [])[:3]
        if financial_topic:
            context_parts.append(f"Financial Topic: {financial_topic}")
        if financial_key_findings:
            context_parts.append(f"Financial Insights: {'; '.join(financial_key_findings)}")
        
        return "\n".join(context_parts)
    
    def _generate_fallback_title(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> str:
        """Generate fallback title when LLM fails"""
        
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
        """Generate comprehensive executive summary using LLM"""
        
        context = self._prepare_summary_context(medical_findings, financial_findings)
        
        prompt = f"""
Create a comprehensive executive summary for a professional research report:

{context}

Requirements:
- 3-4 paragraphs maximum
- Executive-level audience
- Highlight key findings and implications
- Identify cross-domain insights
- Professional, engaging tone
- Include quality indicators
- Focus on actionable insights

Structure:
Paragraph 1: Research scope and methodology
Paragraph 2: Key medical findings and implications
Paragraph 3: Key financial findings and implications
Paragraph 4: Cross-domain insights and recommendations
"""
        
        try:
            response = self.model.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            self.logger.error(f"LLM executive summary generation failed: {e}")
            return self._generate_fallback_summary(medical_findings, financial_findings)
    
    def _prepare_summary_context(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> str:
        """Prepare comprehensive context for executive summary"""
        
        context_parts = []
        
        total_papers = len(medical_findings.get("research_papers", [])) + len(financial_findings.get("research_papers", []))
        context_parts.append(f"RESEARCH SCOPE: Analysis of {total_papers} academic papers")
        
        if medical_findings.get("key_findings"):
            context_parts.append("MEDICAL FINDINGS:")
            context_parts.extend([f"- {finding}" for finding in medical_findings["key_findings"][:5]])
            
            clinical_insights = medical_findings.get("clinical_insights", {})
            if clinical_insights.get("trials"):
                context_parts.append(f"- Clinical trials: {len(clinical_insights['trials'])} studies analyzed")
            
            drug_interactions = medical_findings.get("drug_interactions", {})
            if drug_interactions.get("interactions"):
                context_parts.append(f"- Drug interactions: {len(drug_interactions['interactions'])} identified")
            
            context_parts.append("")
        
        if financial_findings.get("key_findings"):
            context_parts.append("FINANCIAL FINDINGS:")
            context_parts.extend([f"- {finding}" for finding in financial_findings["key_findings"][:5]])
            
            market_analysis = financial_findings.get("market_analysis", {})
            if market_analysis.get("opportunities"):
                context_parts.append(f"- Market opportunities: {len(market_analysis['opportunities'])} identified")
            
            risk_assessment = financial_findings.get("risk_assessment", {})
            if risk_assessment.get("risks"):
                context_parts.append(f"- Risk factors: {len(risk_assessment['risks'])} assessed")
            
            context_parts.append("")
        
        medical_quality = medical_findings.get("quality_indicators", {})
        financial_quality = financial_findings.get("quality_indicators", {})
        
        if medical_quality or financial_quality:
            context_parts.append("QUALITY INDICATORS:")
            if medical_quality.get("avg_relevance"):
                context_parts.append(f"- Medical research quality: {medical_quality['avg_relevance']:.2f}/1.0")
            if financial_quality.get("avg_relevance"):
                context_parts.append(f"- Financial research quality: {financial_quality['avg_relevance']:.2f}/1.0")
        
        return "\n".join(context_parts)
    
    def _generate_fallback_summary(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> str:
        """Generate fallback summary when LLM fails"""
        
        summary_parts = []
        
        if medical_findings.get("key_findings"):
            medical_summary = f"Medical research identified {len(medical_findings['key_findings'])} key findings"
            summary_parts.append(medical_summary)
        
        if financial_findings.get("key_findings"):
            financial_summary = f"Financial analysis revealed {len(financial_findings['key_findings'])} key insights"
            summary_parts.append(financial_summary)
        
        total_papers = len(medical_findings.get("research_papers", [])) + len(financial_findings.get("research_papers", []))
        if total_papers > 0:
            summary_parts.append(f"Analysis based on {total_papers} research papers")
        
        return ". ".join(summary_parts) + "."
    
    def _format_medical_section(self, medical_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Format medical research section with LLM-enhanced content"""
        
        enhanced_content = self._enhance_medical_content(medical_findings)
        
        return {
            "narrative_summary": enhanced_content.get("narrative_summary", ""),
            "key_findings": medical_findings.get("key_findings", []),
            "clinical_insights": medical_findings.get("clinical_insights", {}),
            "drug_interactions": medical_findings.get("drug_interactions", {}),
            "quality_score": medical_findings.get("quality_indicators", {}).get("avg_relevance", 0),
            "source_count": len(medical_findings.get("research_papers", [])),
            "specializations": medical_findings.get("medical_specializations", []),
            "research_implications": enhanced_content.get("research_implications", []),
            "clinical_significance": enhanced_content.get("clinical_significance", "")
        }
    
    def _enhance_medical_content(self, medical_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance medical content using LLM"""
        
        context = self._prepare_medical_context(medical_findings)
        
        prompt = f"""
Create enhanced content for the medical research section of a professional report:

{context}

Generate the following components:

1. NARRATIVE SUMMARY - A compelling 2-3 sentence summary of medical findings
2. RESEARCH IMPLICATIONS - 3-4 bullet points on research implications
3. CLINICAL SIGNIFICANCE - 1-2 sentences on clinical significance

Format your response as:

NARRATIVE SUMMARY:
[2-3 sentences]

RESEARCH IMPLICATIONS:
- [Implication 1]
- [Implication 2]
- [Implication 3]

CLINICAL SIGNIFICANCE:
[1-2 sentences]
"""
        
        try:
            response = self.model.invoke(prompt)
            return self._parse_medical_enhancement(response.content)
        except Exception as e:
            self.logger.error(f"Medical content enhancement failed: {e}")
            return self._generate_fallback_medical_content(medical_findings)
    
    def _prepare_medical_context(self, medical_findings: Dict[str, Any]) -> str:
        """Prepare medical context for enhancement"""
        
        context_parts = []
        
        topic = medical_findings.get("research_topic", "")
        if topic:
            context_parts.append(f"RESEARCH TOPIC: {topic}")
        
        key_findings = medical_findings.get("key_findings", [])
        if key_findings:
            context_parts.append("KEY FINDINGS:")
            context_parts.extend([f"- {finding}" for finding in key_findings[:5]])
        
        clinical_insights = medical_findings.get("clinical_insights", {})
        if clinical_insights.get("trials"):
            context_parts.append(f"CLINICAL TRIALS: {len(clinical_insights['trials'])} studies")
        
        drug_interactions = medical_findings.get("drug_interactions", {})
        if drug_interactions.get("interactions"):
            context_parts.append(f"DRUG INTERACTIONS: {len(drug_interactions['interactions'])} identified")
        
        quality = medical_findings.get("quality_indicators", {})
        if quality.get("avg_relevance"):
            context_parts.append(f"RESEARCH QUALITY: {quality['avg_relevance']:.2f}/1.0")
        
        return "\n".join(context_parts)
    
    def _parse_medical_enhancement(self, enhancement_text: str) -> Dict[str, Any]:
        """Parse medical enhancement response"""
        
        result = {
            "narrative_summary": "",
            "research_implications": [],
            "clinical_significance": ""
        }
        
        current_section = None
        
        for line in enhancement_text.split('\n'):
            line = line.strip()
            
            if line.startswith('NARRATIVE SUMMARY:'):
                current_section = "narrative_summary"
            elif line.startswith('RESEARCH IMPLICATIONS:'):
                current_section = "research_implications"
            elif line.startswith('CLINICAL SIGNIFICANCE:'):
                current_section = "clinical_significance"
            elif line.startswith('- ') and current_section == "research_implications":
                result["research_implications"].append(line[2:])
            elif line and current_section in ["narrative_summary", "clinical_significance"]:
                if result[current_section]:
                    result[current_section] += " " + line
                else:
                    result[current_section] = line
        
        return result
    
    def _generate_fallback_medical_content(self, medical_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback medical content"""
        
        key_findings = medical_findings.get("key_findings", [])
        
        return {
            "narrative_summary": f"Medical research analysis identified {len(key_findings)} key findings with clinical implications.",
            "research_implications": [
                "Clinical trial data supports further investigation",
                "Drug interaction profiles require monitoring",
                "Safety considerations identified for patient populations"
            ][:len(key_findings)],
            "clinical_significance": "Research findings provide important insights for clinical practice and patient care."
        }
    
    def _format_financial_section(self, financial_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Format financial research section with LLM-enhanced content"""
        
        enhanced_content = self._enhance_financial_content(financial_findings)
        
        return {
            "narrative_summary": enhanced_content.get("narrative_summary", ""),
            "key_findings": financial_findings.get("key_findings", []),
            "market_analysis": financial_findings.get("market_analysis", {}),
            "risk_assessment": financial_findings.get("risk_assessment", {}),
            "economic_indicators": financial_findings.get("economic_indicators", {}),
            "quality_score": financial_findings.get("quality_indicators", {}).get("avg_relevance", 0),
            "source_count": len(financial_findings.get("research_papers", [])),
            "specializations": financial_findings.get("financial_specializations", []),
            "investment_insights": enhanced_content.get("investment_insights", []),
            "market_significance": enhanced_content.get("market_significance", "")
        }
    
    def _enhance_financial_content(self, financial_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance financial content using LLM"""
        
        context = self._prepare_financial_context(financial_findings)
        
        prompt = f"""
Create enhanced content for the financial research section of a professional report:

{context}

Generate the following components:

1. NARRATIVE SUMMARY - A compelling 2-3 sentence summary of financial findings
2. INVESTMENT INSIGHTS - 3-4 bullet points on investment insights
3. MARKET SIGNIFICANCE - 1-2 sentences on market significance

Format your response as:

NARRATIVE SUMMARY:
[2-3 sentences]

INVESTMENT INSIGHTS:
- [Insight 1]
- [Insight 2]
- [Insight 3]

MARKET SIGNIFICANCE:
[1-2 sentences]
"""
        
        try:
            response = self.model.invoke(prompt)
            return self._parse_financial_enhancement(response.content)
        except Exception as e:
            self.logger.error(f"Financial content enhancement failed: {e}")
            return self._generate_fallback_financial_content(financial_findings)
    
    def _prepare_financial_context(self, financial_findings: Dict[str, Any]) -> str:
        """Prepare financial context for enhancement"""
        
        context_parts = []
        
        topic = financial_findings.get("research_topic", "")
        if topic:
            context_parts.append(f"RESEARCH TOPIC: {topic}")
        
        key_findings = financial_findings.get("key_findings", [])
        if key_findings:
            context_parts.append("KEY FINDINGS:")
            context_parts.extend([f"- {finding}" for finding in key_findings[:5]])
        
        market_analysis = financial_findings.get("market_analysis", {})
        if market_analysis.get("opportunities"):
            context_parts.append(f"MARKET OPPORTUNITIES: {len(market_analysis['opportunities'])} identified")
        
        risk_assessment = financial_findings.get("risk_assessment", {})
        if risk_assessment.get("risks"):
            context_parts.append(f"RISK FACTORS: {len(risk_assessment['risks'])} assessed")
        
        quality = financial_findings.get("quality_indicators", {})
        if quality.get("avg_relevance"):
            context_parts.append(f"RESEARCH QUALITY: {quality['avg_relevance']:.2f}/1.0")
        
        return "\n".join(context_parts)
    
    def _parse_financial_enhancement(self, enhancement_text: str) -> Dict[str, Any]:
        """Parse financial enhancement response"""
        
        result = {
            "narrative_summary": "",
            "investment_insights": [],
            "market_significance": ""
        }
        
        current_section = None
        
        for line in enhancement_text.split('\n'):
            line = line.strip()
            
            if line.startswith('NARRATIVE SUMMARY:'):
                current_section = "narrative_summary"
            elif line.startswith('INVESTMENT INSIGHTS:'):
                current_section = "investment_insights"
            elif line.startswith('MARKET SIGNIFICANCE:'):
                current_section = "market_significance"
            elif line.startswith('- ') and current_section == "investment_insights":
                result["investment_insights"].append(line[2:])
            elif line and current_section in ["narrative_summary", "market_significance"]:
                if result[current_section]:
                    result[current_section] += " " + line
                else:
                    result[current_section] = line
        
        return result
    
    def _generate_fallback_financial_content(self, financial_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback financial content"""
        
        key_findings = financial_findings.get("key_findings", [])
        
        return {
            "narrative_summary": f"Financial research analysis identified {len(key_findings)} key insights with market implications.",
            "investment_insights": [
                "Market opportunities present investment potential",
                "Risk assessment highlights key considerations",
                "Economic indicators support strategic planning"
            ][:len(key_findings)],
            "market_significance": "Research findings provide valuable insights for investment strategy and market positioning."
        }
    
    def _generate_cross_domain_analysis(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive cross-domain analysis using LLM"""
        
        context = self._prepare_cross_domain_context(medical_findings, financial_findings)
        
        prompt = f"""
Perform comprehensive cross-domain analysis between medical and financial research findings:

{context}

Analyze the following aspects:

1. SYNERGIES - Identify meaningful connections and synergies between medical and financial findings
2. CONFLICTS - Identify potential conflicts or contradictions between domains
3. INVESTMENT IMPLICATIONS - Analyze how medical findings impact investment decisions
4. REGULATORY CONSIDERATIONS - Identify overlapping regulatory requirements

Format your response as:

SYNERGIES:
- [Synergy 1]
- [Synergy 2]

CONFLICTS:
- [Conflict 1]
- [Conflict 2]

INVESTMENT IMPLICATIONS:
- [Implication 1]
- [Implication 2]

REGULATORY CONSIDERATIONS:
- [Consideration 1]
- [Consideration 2]
"""
        
        try:
            response = self.model.invoke(prompt)
            return self._parse_cross_domain_analysis(response.content)
        except Exception as e:
            self.logger.error(f"LLM cross-domain analysis failed: {e}")
            return self._generate_fallback_cross_domain_analysis(medical_findings, financial_findings)
    
    def _prepare_cross_domain_context(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> str:
        """Prepare context for cross-domain analysis"""
        
        context_parts = []
        
        if medical_findings.get("key_findings"):
            context_parts.append("MEDICAL RESEARCH FINDINGS:")
            context_parts.extend([f"- {finding}" for finding in medical_findings["key_findings"][:5]])
            
            clinical_insights = medical_findings.get("clinical_insights", {})
            if clinical_insights.get("trials"):
                context_parts.append(f"- Clinical trials: {len(clinical_insights['trials'])} studies")
            
            drug_interactions = medical_findings.get("drug_interactions", {})
            if drug_interactions.get("safety_considerations"):
                context_parts.append(f"- Safety considerations: {len(drug_interactions['safety_considerations'])}")
            
            context_parts.append("")
        
        if financial_findings.get("key_findings"):
            context_parts.append("FINANCIAL RESEARCH FINDINGS:")
            context_parts.extend([f"- {finding}" for finding in financial_findings["key_findings"][:5]])
            
            market_analysis = financial_findings.get("market_analysis", {})
            if market_analysis.get("opportunities"):
                context_parts.append(f"- Market opportunities: {len(market_analysis['opportunities'])}")
            
            risk_assessment = financial_findings.get("risk_assessment", {})
            if risk_assessment.get("risks"):
                context_parts.append(f"- Risk factors: {len(risk_assessment['risks'])}")
            
            context_parts.append("")
        
        medical_topic = medical_findings.get("research_topic", "")
        financial_topic = financial_findings.get("research_topic", "")
        if medical_topic or financial_topic:
            context_parts.append("RESEARCH TOPICS:")
            if medical_topic:
                context_parts.append(f"- Medical: {medical_topic}")
            if financial_topic:
                context_parts.append(f"- Financial: {financial_topic}")
        
        return "\n".join(context_parts)
    
    def _parse_cross_domain_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse LLM cross-domain analysis response"""
        
        result = {
            "synergies": [],
            "conflicts": [],
            "investment_implications": [],
            "regulatory_considerations": []
        }
        
        current_section = None
        
        for line in analysis_text.split('\n'):
            line = line.strip()
            
            if line.startswith('SYNERGIES:'):
                current_section = "synergies"
            elif line.startswith('CONFLICTS:'):
                current_section = "conflicts"
            elif line.startswith('INVESTMENT IMPLICATIONS:'):
                current_section = "investment_implications"
            elif line.startswith('REGULATORY CONSIDERATIONS:'):
                current_section = "regulatory_considerations"
            elif line.startswith('- ') and current_section:
                result[current_section].append(line[2:])
        
        return result
    
    def _generate_fallback_cross_domain_analysis(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback cross-domain analysis when LLM fails"""
        
        return {
            "synergies": self._identify_synergies_fallback(medical_findings, financial_findings),
            "conflicts": self._identify_conflicts_fallback(medical_findings, financial_findings),
            "investment_implications": self._analyze_investment_implications_fallback(medical_findings, financial_findings),
            "regulatory_considerations": self._analyze_regulatory_overlap_fallback(medical_findings, financial_findings)
        }
    
    def _identify_synergies_fallback(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> List[str]:
        """Fallback synergy identification"""
        
        synergies = []
        
        medical_terms = set(medical_findings.get("key_terms", []))
        financial_terms = set(financial_findings.get("key_terms", []))
        
        common_terms = medical_terms.intersection(financial_terms)
        if common_terms:
            synergies.append(f"Common research themes: {', '.join(common_terms)}")
        
        if medical_findings.get("clinical_insights") and financial_findings.get("market_analysis"):
            synergies.append("Healthcare investment opportunities identified from clinical research")
        
        return synergies
    
    def _identify_conflicts_fallback(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> List[str]:
        """Fallback conflict identification"""
        
        conflicts = []
        
        medical_risks = medical_findings.get("drug_interactions", {}).get("safety_considerations", [])
        financial_risks = financial_findings.get("risk_assessment", {}).get("risks", [])
        
        if medical_risks and financial_risks:
            conflicts.append("Potential conflicts between medical safety and financial risk considerations")
        
        return conflicts
    
    def _analyze_investment_implications_fallback(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> List[str]:
        """Fallback investment implications analysis"""
        
        implications = []
        
        if medical_findings.get("key_findings"):
            implications.append("Medical research findings may have investment implications")
        
        if financial_findings.get("market_analysis", {}).get("opportunities"):
            implications.append("Financial analysis identifies healthcare market opportunities")
        
        return implications
    
    def _analyze_regulatory_overlap_fallback(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> List[str]:
        """Fallback regulatory overlap analysis"""
        
        regulatory_items = []
        
        if medical_findings.get("clinical_insights"):
            regulatory_items.append("Medical regulatory compliance required")
        
        if financial_findings.get("risk_assessment", {}).get("regulatory_considerations"):
            regulatory_items.append("Financial regulatory considerations identified")
        
        return regulatory_items
    
    def _generate_recommendations(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations using LLM"""
        
        context = self._prepare_recommendations_context(medical_findings, financial_findings)
        
        prompt = f"""
Generate actionable, strategic recommendations based on this research analysis:

{context}

Requirements:
- Strategic and actionable recommendations
- Consider both medical and financial implications
- Address identified risks and opportunities
- Executive-level audience
- 5-8 specific recommendations
- Prioritize by impact and feasibility

Format each recommendation as:
- [Recommendation]: [Brief justification]
"""
        
        try:
            response = self.model.invoke(prompt)
            return self._parse_recommendations(response.content)
        except Exception as e:
            self.logger.error(f"LLM recommendations generation failed: {e}")
            return self._generate_fallback_recommendations(medical_findings, financial_findings)
    
    def _prepare_recommendations_context(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> str:
        """Prepare context for recommendations generation"""
        
        context_parts = []
        
        context_parts.append("KEY FINDINGS SUMMARY:")
        
        medical_key_findings = medical_findings.get("key_findings", [])
        if medical_key_findings:
            context_parts.append("Medical:")
            context_parts.extend([f"- {finding}" for finding in medical_key_findings[:3]])
        
        financial_key_findings = financial_findings.get("key_findings", [])
        if financial_key_findings:
            context_parts.append("Financial:")
            context_parts.extend([f"- {finding}" for finding in financial_key_findings[:3]])
        
        context_parts.append("")
        context_parts.append("OPPORTUNITIES AND RISKS:")
        
        clinical_insights = medical_findings.get("clinical_insights", {})
        if clinical_insights.get("trials"):
            context_parts.append(f"- Clinical opportunities: {len(clinical_insights['trials'])} potential studies")
        
        market_analysis = financial_findings.get("market_analysis", {})
        if market_analysis.get("opportunities"):
            context_parts.append(f"- Market opportunities: {len(market_analysis['opportunities'])} identified")
        
        medical_risks = medical_findings.get("drug_interactions", {}).get("safety_considerations", [])
        financial_risks = financial_findings.get("risk_assessment", {}).get("risks", [])
        
        if medical_risks:
            context_parts.append(f"- Medical risks: {len(medical_risks)} safety considerations")
        if financial_risks:
            context_parts.append(f"- Financial risks: {len(financial_risks)} risk factors")
        
        context_parts.append("")
        context_parts.append("RESEARCH QUALITY:")
        
        medical_quality = medical_findings.get("quality_indicators", {})
        financial_quality = financial_findings.get("quality_indicators", {})
        
        if medical_quality.get("avg_relevance"):
            context_parts.append(f"- Medical research quality: {medical_quality['avg_relevance']:.2f}/1.0")
        if financial_quality.get("avg_relevance"):
            context_parts.append(f"- Financial research quality: {financial_quality['avg_relevance']:.2f}/1.0")
        
        return "\n".join(context_parts)
    
    def _parse_recommendations(self, recommendations_text: str) -> List[str]:
        """Parse LLM recommendations response"""
        
        recommendations = []
        
        for line in recommendations_text.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                recommendations.append(line[2:])
        
        return recommendations
    
    def _generate_fallback_recommendations(self, medical_findings: Dict[str, Any], financial_findings: Dict[str, Any]) -> List[str]:
        """Generate fallback recommendations when LLM fails"""
        
        recommendations = []
        
        if medical_findings.get("clinical_insights"):
            recommendations.append("Consider clinical trial implications for investment decisions")
        
        if financial_findings.get("market_analysis", {}).get("opportunities"):
            recommendations.append("Explore identified market opportunities")
        
        medical_risks = medical_findings.get("drug_interactions", {}).get("safety_considerations", [])
        financial_risks = financial_findings.get("risk_assessment", {}).get("risks", [])
        
        if medical_risks or financial_risks:
            recommendations.append("Implement comprehensive risk management strategies")
        
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