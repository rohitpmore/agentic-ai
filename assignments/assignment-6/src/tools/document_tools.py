import os
from typing import Dict, Any, Optional
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import logging
from datetime import datetime


class DocumentTool:
    """Tool for creating professional documents"""
    
    def __init__(self, settings):
        self.settings = settings
        self.logger = logging.getLogger("document_tool")
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom document styles"""
        
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6
        ))
        
    def create_document(
        self,
        structure: Dict[str, Any],
        format: str = "pdf",
        template: str = "professional_report"
    ) -> str:
        """Create document from structure"""
        
        try:
            # Ensure output directory exists
            self.settings.ensure_output_directory()
            
            # Generate filename
            filename = self._generate_filename(structure.get("title", "report"), format)
            file_path = os.path.join(self.settings.output_directory, filename)
            
            if format.lower() == "pdf":
                self._create_pdf(structure, file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Document created: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Document creation failed: {e}")
            raise
    
    def _generate_filename(self, title: str, format: str) -> str:
        """Generate unique filename"""
        
        # Clean title for filename
        clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title.replace(' ', '_')[:50]
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{clean_title}_{timestamp}.{format}"
    
    def _create_pdf(self, structure: Dict[str, Any], file_path: str):
        """Create PDF document"""
        
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        story = []
        
        # Title
        title = structure.get("title", "Research Report")
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        executive_summary = structure.get("executive_summary", "")
        if executive_summary:
            story.append(Paragraph("Executive Summary", self.styles['CustomHeading']))
            story.append(Paragraph(executive_summary, self.styles['CustomBody']))
            story.append(Spacer(1, 15))
        
        # Sections
        sections = structure.get("sections", [])
        for section in sections:
            section_title = section.get("title", "")
            section_content = section.get("content", {})
            
            # Section title
            story.append(Paragraph(section_title, self.styles['CustomHeading']))
            
            # Section content
            if isinstance(section_content, dict):
                self._add_dict_content(story, section_content)
            else:
                story.append(Paragraph(str(section_content), self.styles['CustomBody']))
            
            story.append(Spacer(1, 15))
        
        # Appendices
        appendices = structure.get("appendices", [])
        if appendices:
            story.append(Paragraph("Appendices", self.styles['CustomHeading']))
            for appendix in appendices:
                appendix_title = appendix.get("title", "")
                story.append(Paragraph(appendix_title, self.styles['Heading3']))
                
                appendix_content = appendix.get("content", {})
                if isinstance(appendix_content, dict):
                    self._add_dict_content(story, appendix_content)
                else:
                    story.append(Paragraph(str(appendix_content), self.styles['CustomBody']))
                
                story.append(Spacer(1, 10))
        
        # Build document
        doc.build(story)
    
    def _add_dict_content(self, story: list, content: Dict[str, Any]):
        """Add dictionary content to story"""
        
        for key, value in content.items():
            if isinstance(value, list):
                story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b>", self.styles['CustomBody']))
                for item in value:
                    story.append(Paragraph(f"• {item}", self.styles['CustomBody']))
            elif isinstance(value, dict):
                story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b>", self.styles['CustomBody']))
                for subkey, subvalue in value.items():
                    story.append(Paragraph(f"  {subkey}: {subvalue}", self.styles['CustomBody']))
            else:
                story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", self.styles['CustomBody']))
        
        story.append(Spacer(1, 6))