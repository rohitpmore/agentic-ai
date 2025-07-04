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