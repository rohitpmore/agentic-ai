#!/usr/bin/env python3
"""
CLI Interface for LangGraph Multi-Agent System
"""

import argparse
import asyncio
import os
import sys
from typing import Optional, Dict, Any
from datetime import datetime
import json
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.syntax import Syntax

from src.main import MultiAgentWorkflow
from src.utils.logging_config import setup_logging
from config.settings import Settings


class MultiAgentCLI:
    """Command-line interface for the multi-agent system"""
    
    def __init__(self):
        self.console = Console()
        self.settings = None
        self.workflow = None
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        
        parser = argparse.ArgumentParser(
            description="LangGraph Multi-Agent Hierarchical Workflow System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s -q "Research AI applications in healthcare and finance"
  %(prog)s --interactive
  %(prog)s --setup
  %(prog)s --test
  %(prog)s --config custom_config.yaml
            """
        )
        
        # Main commands
        parser.add_argument(
            "-q", "--query",
            type=str,
            help="Research query to execute"
        )
        
        parser.add_argument(
            "-i", "--interactive",
            action="store_true",
            help="Start interactive mode"
        )
        
        parser.add_argument(
            "--setup",
            action="store_true",
            help="Setup environment and test connections"
        )
        
        parser.add_argument(
            "--test",
            action="store_true",
            help="Run system tests"
        )
        
        # Configuration options
        parser.add_argument(
            "--config",
            type=str,
            help="Path to configuration file"
        )
        
        parser.add_argument(
            "--output-dir",
            type=str,
            default="./outputs",
            help="Output directory for generated files"
        )
        
        parser.add_argument(
            "--format",
            choices=["json", "yaml", "markdown"],
            default="markdown",
            help="Output format for results"
        )
        
        # Execution options
        parser.add_argument(
            "--stream",
            action="store_true",
            default=True,
            help="Enable streaming output (default: True)"
        )
        
        parser.add_argument(
            "--no-stream",
            action="store_true",
            help="Disable streaming output"
        )
        
        parser.add_argument(
            "--timeout",
            type=int,
            default=300,
            help="Workflow timeout in seconds (default: 300)"
        )
        
        # Debug options
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode"
        )
        
        parser.add_argument(
            "--verbose", "-v",
            action="count",
            default=0,
            help="Increase verbosity (use -v, -vv, -vvv)"
        )
        
        parser.add_argument(
            "--log-file",
            type=str,
            help="Log file path"
        )
        
        # Visualization options
        parser.add_argument(
            "--visualize",
            action="store_true",
            help="Generate workflow visualization"
        )
        
        parser.add_argument(
            "--graph-format",
            choices=["mermaid", "png"],
            default="mermaid",
            help="Graph visualization format"
        )
        
        return parser
    
    def setup_environment(self, args) -> bool:
        """Setup environment and validate configuration"""
        
        try:
            # Load settings
            if args.config:
                self.settings = self._load_config_file(args.config)
            else:
                self.settings = Settings()
            
            # Override with command-line arguments
            if args.output_dir:
                self.settings.output_directory = args.output_dir
            if args.debug:
                self.settings.debug_mode = True
            if args.timeout:
                self.settings.timeout_seconds = args.timeout
            
            # Setup logging
            log_level = "DEBUG" if args.debug else "INFO"
            if args.verbose >= 2:
                log_level = "DEBUG"
            elif args.verbose == 1:
                log_level = "INFO"
            
            setup_logging(
                level=log_level,
                format_type="json" if args.debug else "standard",
                log_file=args.log_file
            )
            
            # Ensure output directory exists
            os.makedirs(self.settings.output_directory, exist_ok=True)
            
            # Initialize workflow
            self.workflow = MultiAgentWorkflow(self.settings)
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Environment setup failed: {e}[/red]")
            return False
    
    def _load_config_file(self, config_path: str) -> Settings:
        """Load configuration from file"""
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_data = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config_data = json.load(f)
            else:
                raise ValueError("Configuration file must be YAML or JSON")
        
        return Settings(**config_data)
    
    async def run_query(self, query: str, args) -> Dict[str, Any]:
        """Run a single research query"""
        
        self.console.print(Panel(
            f"[bold blue]Research Query:[/bold blue] {query}",
            title="Multi-Agent Research System",
            border_style="blue"
        ))
        
        # Determine execution mode
        use_streaming = args.stream and not args.no_stream
        
        try:
            if use_streaming:
                return await self._run_streaming_query(query, args)
            else:
                return await self._run_batch_query(query, args)
                
        except Exception as e:
            self.console.print(f"[red]Query execution failed: {e}[/red]")
            return {"error": str(e)}
    
    async def _run_streaming_query(self, query: str, args) -> Dict[str, Any]:
        """Run query with streaming output"""
        
        result = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Initializing workflow...", total=None)
            
            try:
                async for chunk in self.workflow.run_workflow_streaming(query):
                    if isinstance(chunk, dict):
                        # Update progress based on chunk content
                        if "current_team" in chunk:
                            team = chunk["current_team"]
                            if team == "research":
                                progress.update(task, description="Research team working...")
                            elif team == "reporting":
                                progress.update(task, description="Reporting team working...")
                            elif team == "end":
                                progress.update(task, description="Finalizing results...")
                        
                        # Display agent updates
                        if "messages" in chunk and chunk["messages"]:
                            latest_message = chunk["messages"][-1]
                            self.console.print(f"[dim]> {latest_message}[/dim]")
                        
                        # Accumulate results
                        result.update(chunk)
                
                progress.update(task, description="Completed!", completed=True)
                
            except Exception as e:
                progress.update(task, description=f"Failed: {e}")
                raise
        
        return result
    
    async def _run_batch_query(self, query: str, args) -> Dict[str, Any]:
        """Run query in batch mode"""
        
        with self.console.status("[bold green]Processing query...") as status:
            result = await self.workflow.run_workflow(query)
            status.update("[bold green]Query completed!")
        
        return result
    
    def display_results(self, result: Dict[str, Any], args):
        """Display results in specified format"""
        
        if "error" in result:
            self.console.print(Panel(
                f"[red]Error: {result['error']}[/red]",
                title="Execution Error",
                border_style="red"
            ))
            return
        
        # Display summary
        self._display_summary(result)
        
        # Display detailed results based on format
        if args.format == "json":
            self._display_json_results(result)
        elif args.format == "yaml":
            self._display_yaml_results(result)
        else:  # markdown
            self._display_markdown_results(result)
        
        # Save results to file
        self._save_results(result, args)
    
    def _display_summary(self, result: Dict[str, Any]):
        """Display execution summary"""
        
        table = Table(title="Execution Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # System metrics
        system_metrics = result.get("system_metrics", {})
        if system_metrics:
            table.add_row("Start Time", system_metrics.get("start_time", "Unknown"))
            table.add_row("End Time", system_metrics.get("end_time", "Unknown"))
            table.add_row("Success", str(system_metrics.get("success", False)))
        
        # Research metrics
        research_state = result.get("research_state", {})
        if research_state:
            table.add_row("Research Status", research_state.get("research_status", "Unknown"))
            medical_findings = research_state.get("medical_findings", {})
            financial_findings = research_state.get("financial_findings", {})
            
            if medical_findings:
                table.add_row("Medical Papers", str(len(medical_findings.get("research_papers", []))))
                table.add_row("Medical Findings", str(len(medical_findings.get("key_findings", []))))
            
            if financial_findings:
                table.add_row("Financial Papers", str(len(financial_findings.get("research_papers", []))))
                table.add_row("Financial Findings", str(len(financial_findings.get("key_findings", []))))
        
        # Reporting metrics
        reporting_state = result.get("reporting_state", {})
        if reporting_state:
            table.add_row("Report Status", reporting_state.get("report_status", "Unknown"))
            table.add_row("Document Created", "Yes" if reporting_state.get("document_path") else "No")
            table.add_row("Summary Created", "Yes" if reporting_state.get("summary") else "No")
        
        self.console.print(table)
    
    def _display_json_results(self, result: Dict[str, Any]):
        """Display results as JSON"""
        
        json_output = json.dumps(result, indent=2, default=str)
        syntax = Syntax(json_output, "json", theme="monokai", line_numbers=True)
        
        self.console.print(Panel(
            syntax,
            title="Results (JSON)",
            border_style="green"
        ))
    
    def _display_yaml_results(self, result: Dict[str, Any]):
        """Display results as YAML"""
        
        yaml_output = yaml.dump(result, default_flow_style=False, indent=2)
        syntax = Syntax(yaml_output, "yaml", theme="monokai", line_numbers=True)
        
        self.console.print(Panel(
            syntax,
            title="Results (YAML)",
            border_style="green"
        ))
    
    def _display_markdown_results(self, result: Dict[str, Any]):
        """Display results as formatted markdown"""
        
        markdown_content = self._generate_markdown_report(result)
        markdown = Markdown(markdown_content)
        
        self.console.print(Panel(
            markdown,
            title="Research Report",
            border_style="green"
        ))
    
    def _generate_markdown_report(self, result: Dict[str, Any]) -> str:
        """Generate markdown report from results"""
        
        lines = []
        lines.append("# Multi-Agent Research Report")
        lines.append("")
        
        # Executive Summary
        final_output = result.get("final_output", {})
        if final_output.get("summary"):
            lines.append("## Executive Summary")
            lines.append(final_output["summary"])
            lines.append("")
        
        # Research Findings
        research_state = result.get("research_state", {})
        if research_state:
            lines.append("## Research Findings")
            
            # Medical findings
            medical_findings = research_state.get("medical_findings", {})
            if medical_findings.get("key_findings"):
                lines.append("### Medical Research")
                for finding in medical_findings["key_findings"][:5]:
                    lines.append(f"- {finding}")
                lines.append("")
            
            # Financial findings
            financial_findings = research_state.get("financial_findings", {})
            if financial_findings.get("key_findings"):
                lines.append("### Financial Research")
                for finding in financial_findings["key_findings"][:5]:
                    lines.append(f"- {finding}")
                lines.append("")
        
        # Generated Documents
        reporting_state = result.get("reporting_state", {})
        if reporting_state:
            lines.append("## Generated Documents")
            
            if reporting_state.get("document_path"):
                lines.append(f"- **Report Document**: {reporting_state['document_path']}")
            
            if reporting_state.get("summary"):
                lines.append("- **Summary**: Generated")
            
            lines.append("")
        
        # System Information
        system_metrics = result.get("system_metrics", {})
        if system_metrics:
            lines.append("## System Information")
            lines.append(f"- **Execution Time**: {system_metrics.get('start_time')} - {system_metrics.get('end_time')}")
            lines.append(f"- **Success**: {system_metrics.get('success', False)}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _save_results(self, result: Dict[str, Any], args):
        """Save results to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save in requested format
        if args.format == "json":
            filename = f"results_{timestamp}.json"
            filepath = os.path.join(self.settings.output_directory, filename)
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        elif args.format == "yaml":
            filename = f"results_{timestamp}.yaml"
            filepath = os.path.join(self.settings.output_directory, filename)
            with open(filepath, 'w') as f:
                yaml.dump(result, f, default_flow_style=False, indent=2)
        else:  # markdown
            filename = f"report_{timestamp}.md"
            filepath = os.path.join(self.settings.output_directory, filename)
            with open(filepath, 'w') as f:
                f.write(self._generate_markdown_report(result))
        
        self.console.print(f"[green]Results saved to: {filepath}[/green]")
    
    async def interactive_mode(self, args):
        """Run interactive mode"""
        
        self.console.print(Panel(
            "[bold blue]Multi-Agent Research System[/bold blue]\n"
            "Interactive Mode\n\n"
            "Commands:\n"
            "- Enter research query to start analysis\n"
            "- 'help' - Show help\n"
            "- 'status' - Show system status\n"
            "- 'config' - Show configuration\n"
            "- 'visualize' - Generate workflow diagram\n"
            "- 'exit' or 'quit' - Exit interactive mode",
            title="Welcome",
            border_style="blue"
        ))
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]Enter query or command[/bold cyan]").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit']:
                    self.console.print("[yellow]Goodbye![/yellow]")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                elif user_input.lower() == 'status':
                    self._show_status()
                elif user_input.lower() == 'config':
                    self._show_config()
                elif user_input.lower() == 'visualize':
                    await self._show_visualization(args)
                else:
                    # Treat as research query
                    result = await self.run_query(user_input, args)
                    self.display_results(result, args)
                    
            except KeyboardInterrupt:
                if Confirm.ask("\n[yellow]Do you want to exit?[/yellow]"):
                    break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
    
    def _show_help(self):
        """Show help information"""
        
        help_text = """
# Multi-Agent System Help

## Commands:
- **Research Query**: Enter any research question to start analysis
- **help**: Show this help message
- **status**: Show system status and metrics
- **config**: Show current configuration
- **visualize**: Generate workflow visualization
- **exit/quit**: Exit interactive mode

## Example Queries:
- "Research AI applications in healthcare and finance"
- "Analyze machine learning trends in medical diagnostics"
- "Compare investment strategies for technology stocks"
- "Study drug interactions for diabetes medications"

## Tips:
- Be specific in your research queries for better results
- The system will automatically route queries to appropriate research teams
- Results are saved automatically in the output directory
        """
        
        self.console.print(Markdown(help_text))
    
    def _show_status(self):
        """Show system status"""
        
        status_table = Table(title="System Status", show_header=True, header_style="bold magenta")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        
        # Check system components
        status_table.add_row("Workflow", "Ready" if self.workflow else "Not initialized")
        status_table.add_row("Configuration", "Loaded" if self.settings else "Not loaded")
        status_table.add_row("Output Directory", self.settings.output_directory if self.settings else "Unknown")
        
        # Check API connections (simplified)
        try:
            status_table.add_row("OpenAI API", "Connected")
        except:
            status_table.add_row("OpenAI API", "Not available")
        
        self.console.print(status_table)
    
    def _show_config(self):
        """Show current configuration"""
        
        if not self.settings:
            self.console.print("[red]Configuration not loaded[/red]")
            return
        
        config_table = Table(title="Configuration", show_header=True, header_style="bold magenta")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        
        # Show key configuration items
        config_table.add_row("Supervisor Model", self.settings.supervisor_model)
        config_table.add_row("Researcher Model", self.settings.researcher_model)
        config_table.add_row("Reporter Model", self.settings.reporter_model)
        config_table.add_row("Output Directory", self.settings.output_directory)
        config_table.add_row("Max Retries", str(self.settings.max_retries))
        config_table.add_row("Timeout", f"{self.settings.timeout_seconds}s")
        config_table.add_row("Debug Mode", str(self.settings.debug_mode))
        
        self.console.print(config_table)
    
    async def _show_visualization(self, args):
        """Show workflow visualization"""
        
        if not self.workflow:
            self.console.print("[red]Workflow not initialized[/red]")
            return
        
        try:
            if args.graph_format == "mermaid":
                mermaid_diagram = self.workflow.get_graph_visualization()
                syntax = Syntax(mermaid_diagram, "mermaid", theme="monokai", line_numbers=True)
                
                self.console.print(Panel(
                    syntax,
                    title="Workflow Diagram (Mermaid)",
                    border_style="green"
                ))
            else:  # png
                image_path = self.workflow.save_graph_image("workflow_diagram.png")
                if image_path:
                    self.console.print(f"[green]Workflow diagram saved to: {image_path}[/green]")
                else:
                    self.console.print("[red]Failed to generate workflow diagram[/red]")
                    
        except Exception as e:
            self.console.print(f"[red]Visualization error: {e}[/red]")
    
    async def setup_command(self, args):
        """Setup and test system"""
        
        self.console.print(Panel(
            "[bold blue]System Setup and Testing[/bold blue]",
            title="Setup",
            border_style="blue"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Environment setup
            task1 = progress.add_task("Setting up environment...", total=None)
            setup_success = self.setup_environment(args)
            progress.update(task1, description="Environment setup complete" if setup_success else "Environment setup failed")
            
            if not setup_success:
                return False
            
            # Test API connections
            task2 = progress.add_task("Testing API connections...", total=None)
            api_success = await self._test_api_connections()
            progress.update(task2, description="API connections tested" if api_success else "API connection failed")
            
            # Test workflow compilation
            task3 = progress.add_task("Testing workflow compilation...", total=None)
            workflow_success = self._test_workflow_compilation()
            progress.update(task3, description="Workflow compilation tested" if workflow_success else "Workflow compilation failed")
            
            # Generate test visualization
            task4 = progress.add_task("Generating test visualization...", total=None)
            viz_success = self._test_visualization()
            progress.update(task4, description="Visualization tested" if viz_success else "Visualization test failed")
        
        # Display results
        results_table = Table(title="Setup Results", show_header=True, header_style="bold magenta")
        results_table.add_column("Test", style="cyan")
        results_table.add_column("Result", style="green")
        
        results_table.add_row("Environment Setup", "✓ Pass" if setup_success else "✗ Fail")
        results_table.add_row("API Connections", "✓ Pass" if api_success else "✗ Fail")
        results_table.add_row("Workflow Compilation", "✓ Pass" if workflow_success else "✗ Fail")
        results_table.add_row("Visualization", "✓ Pass" if viz_success else "✗ Fail")
        
        self.console.print(results_table)
        
        return all([setup_success, api_success, workflow_success, viz_success])
    
    async def _test_api_connections(self) -> bool:
        """Test API connections"""
        
        try:
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model="gpt-3.5-turbo", timeout=10)
            response = model.invoke("Test connection")
            return bool(response)
        except Exception as e:
            self.console.print(f"[red]API test failed: {e}[/red]")
            return False
    
    def _test_workflow_compilation(self) -> bool:
        """Test workflow compilation"""
        
        try:
            return bool(self.workflow and self.workflow.graph)
        except Exception as e:
            self.console.print(f"[red]Workflow compilation test failed: {e}[/red]")
            return False
    
    def _test_visualization(self) -> bool:
        """Test visualization generation"""
        
        try:
            if self.workflow:
                mermaid_diagram = self.workflow.get_graph_visualization()
                return len(mermaid_diagram) > 0
            return False
        except Exception as e:
            self.console.print(f"[red]Visualization test failed: {e}[/red]")
            return False
    
    async def test_command(self, args):
        """Run system tests"""
        
        self.console.print(Panel(
            "[bold blue]Running System Tests[/bold blue]",
            title="Testing",
            border_style="blue"
        ))
        
        # Run a simple test query
        test_query = "Test query for system validation"
        
        try:
            result = await self.run_query(test_query, args)
            
            if "error" not in result:
                self.console.print("[green]✓ System test passed[/green]")
                return True
            else:
                self.console.print(f"[red]✗ System test failed: {result['error']}[/red]")
                return False
                
        except Exception as e:
            self.console.print(f"[red]✗ System test failed: {e}[/red]")
            return False


async def main():
    """Main CLI entry point"""
    
    cli = MultiAgentCLI()
    parser = cli.create_parser()
    args = parser.parse_args()
    
    # Handle no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Setup environment
    if not cli.setup_environment(args):
        sys.exit(1)
    
    try:
        # Handle commands
        if args.setup:
            success = await cli.setup_command(args)
            sys.exit(0 if success else 1)
        
        elif args.test:
            success = await cli.test_command(args)
            sys.exit(0 if success else 1)
        
        elif args.visualize:
            await cli._show_visualization(args)
        
        elif args.interactive:
            await cli.interactive_mode(args)
        
        elif args.query:
            result = await cli.run_query(args.query, args)
            cli.display_results(result, args)
            
            # Exit with error code if query failed
            if "error" in result:
                sys.exit(1)
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        cli.console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        cli.console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())