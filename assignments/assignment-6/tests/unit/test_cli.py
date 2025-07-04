"""
CLI Testing Framework
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from io import StringIO
import sys
import os
import tempfile

from src.cli import MultiAgentCLI, main
from src.cli_config import CLIConfig, CLIConfigManager
from config.settings import Settings


class TestMultiAgentCLI:
    @pytest.fixture
    def cli(self):
        return MultiAgentCLI()
    
    @pytest.fixture
    def mock_settings(self):
        return Settings(
            openai_api_key="test-key",
            supervisor_model="gpt-4",
            output_directory="./test_outputs"
        )
    
    def test_parser_creation(self, cli):
        parser = cli.create_parser()
        assert parser.prog is not None
        
        # Test basic arguments
        args = parser.parse_args(["-q", "test query"])
        assert args.query == "test query"
        
        args = parser.parse_args(["--interactive"])
        assert args.interactive is True
        
        args = parser.parse_args(["--setup"])
        assert args.setup is True
    
    @patch('src.cli.Settings')
    @patch('src.cli.MultiAgentWorkflow')
    def test_environment_setup(self, mock_workflow, mock_settings, cli):
        mock_settings.return_value = Mock()
        mock_workflow.return_value = Mock()
        
        # Mock arguments
        args = Mock()
        args.config = None
        args.output_dir = "./test"
        args.debug = False
        args.timeout = 300
        args.verbose = 0
        args.log_file = None
        
        with patch('os.makedirs'):
            success = cli.setup_environment(args)
        
        assert success is True
        assert cli.settings is not None
        assert cli.workflow is not None
    
    @pytest.mark.asyncio
    @patch('src.cli.MultiAgentWorkflow')
    async def test_run_query_streaming(self, mock_workflow_class, cli):
        # Mock workflow
        mock_workflow = Mock()
        
        async def mock_streaming():
            data = [
                {"current_team": "research", "messages": ["Starting research"]},
                {"current_team": "reporting", "messages": ["Creating report"]},
                {"final_output": {"summary": "Test summary"}}
            ]
            for item in data:
                yield item
        
        mock_workflow.run_workflow_streaming = Mock(return_value=mock_streaming())
        
        cli.workflow = mock_workflow
        
        # Mock Rich Progress context manager
        mock_progress = Mock()
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        mock_progress.add_task = Mock(return_value="task1")
        mock_progress.update = Mock()
        
        with patch('src.cli.Progress', return_value=mock_progress):
            with patch.object(cli, 'console') as mock_console:
                # Mock arguments
                args = Mock()
                args.stream = True
                args.no_stream = False
                
                result = await cli.run_query("test query", args)
        
        assert "final_output" in result
        mock_workflow.run_workflow_streaming.assert_called_once_with("test query")
    
    @pytest.mark.asyncio
    @patch('src.cli.MultiAgentWorkflow')
    async def test_run_query_batch(self, mock_workflow_class, cli):
        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.run_workflow = AsyncMock()
        mock_workflow.run_workflow.return_value = {
            "final_output": {"summary": "Test summary"},
            "system_metrics": {"success": True}
        }
        
        cli.workflow = mock_workflow
        
        # Mock the status context manager
        mock_status = Mock()
        mock_status.__enter__ = Mock(return_value=mock_status)
        mock_status.__exit__ = Mock(return_value=None)
        mock_status.update = Mock()
        
        with patch.object(cli, 'console') as mock_console:
            mock_console.status = Mock(return_value=mock_status)
            
            # Mock arguments
            args = Mock()
            args.stream = False
            args.no_stream = True
            
            result = await cli.run_query("test query", args)
        
        assert "final_output" in result
        mock_workflow.run_workflow.assert_called_once_with("test query")
    
    def test_display_results_json(self, cli):
        cli.console = Mock()
        cli.settings = Mock()
        cli.settings.output_directory = "./test"
        
        result = {
            "final_output": {"summary": "Test summary"},
            "system_metrics": {"success": True}
        }
        
        args = Mock()
        args.format = "json"
        
        # Mock file operations
        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=None)
        
        with patch('builtins.open', return_value=mock_file):
            with patch('json.dump'):
                cli.display_results(result, args)
        
        # Verify console.print was called
        assert cli.console.print.called
    
    def test_markdown_report_generation(self, cli):
        result = {
            "final_output": {"summary": "Test executive summary"},
            "research_state": {
                "medical_findings": {"key_findings": ["Medical finding 1", "Medical finding 2"]},
                "financial_findings": {"key_findings": ["Financial finding 1"]}
            },
            "reporting_state": {
                "document_path": "/path/to/document.pdf",
                "summary": "Test summary"
            },
            "system_metrics": {
                "start_time": "2023-01-01T00:00:00",
                "end_time": "2023-01-01T01:00:00",
                "success": True
            }
        }
        
        markdown = cli._generate_markdown_report(result)
        
        assert "# Multi-Agent Research Report" in markdown
        assert "## Executive Summary" in markdown
        assert "Test executive summary" in markdown
        assert "### Medical Research" in markdown
        assert "### Financial Research" in markdown
        assert "Medical finding 1" in markdown
        assert "Financial finding 1" in markdown
    
    def test_error_handling(self, cli):
        cli.console = Mock()
        
        result = {"error": "Test error message"}
        args = Mock()
        
        cli.display_results(result, args)
        
        # Should display error panel
        assert cli.console.print.called
    
    @pytest.mark.asyncio
    async def test_setup_command(self, cli):
        # Mock setup methods
        cli.setup_environment = Mock(return_value=True)
        cli._test_api_connections = AsyncMock(return_value=True)
        cli._test_workflow_compilation = Mock(return_value=True)
        cli._test_visualization = Mock(return_value=True)
        
        # Mock Rich Progress context manager
        mock_progress = Mock()
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        mock_progress.add_task = Mock(return_value="task1")
        mock_progress.update = Mock()
        
        with patch('src.cli.Progress', return_value=mock_progress):
            with patch.object(cli, 'console') as mock_console:
                args = Mock()
                result = await cli.setup_command(args)
        
        assert result is True
        cli.setup_environment.assert_called_once()
        cli._test_api_connections.assert_called_once()
        cli._test_workflow_compilation.assert_called_once()
        cli._test_visualization.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_test_command(self, cli):
        cli.console = Mock()
        cli.run_query = AsyncMock(return_value={"final_output": {"summary": "Test"}})
        
        args = Mock()
        result = await cli.test_command(args)
        
        assert result is True
        cli.run_query.assert_called_once()
    
    def test_show_status(self, cli):
        cli.console = Mock()
        cli.workflow = Mock()
        cli.settings = Mock()
        cli.settings.output_directory = "./test"
        
        cli._show_status()
        
        # Should display status table
        assert cli.console.print.called
    
    def test_show_config(self, cli):
        cli.console = Mock()
        cli.settings = Mock()
        cli.settings.supervisor_model = "gpt-4"
        cli.settings.researcher_model = "gpt-3.5-turbo"
        cli.settings.reporter_model = "gpt-3.5-turbo"
        cli.settings.output_directory = "./test"
        cli.settings.max_retries = 3
        cli.settings.timeout_seconds = 300
        cli.settings.debug_mode = False
        
        cli._show_config()
        
        # Should display config table
        assert cli.console.print.called
    
    @pytest.mark.asyncio
    async def test_show_visualization(self, cli):
        cli.console = Mock()
        cli.workflow = Mock()
        cli.workflow.get_graph_visualization = Mock(return_value="graph TD\n    A --> B")
        
        args = Mock()
        args.graph_format = "mermaid"
        
        await cli._show_visualization(args)
        
        # Should display visualization
        assert cli.console.print.called
        cli.workflow.get_graph_visualization.assert_called_once()


class TestCLIConfig:
    def test_default_config_creation(self):
        config = CLIConfig()
        assert config.use_color is True
        assert config.default_format == "markdown"
        assert config.streaming_enabled is True
    
    def test_config_save_load(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_file = f.name
        
        try:
            # Create config
            config = CLIConfig(use_color=False, verbose_output=True)
            
            # Save to file
            config.save_to_file(config_file)
            
            # Load from file
            loaded_config = CLIConfig.load_from_file(config_file)
            
            assert loaded_config.use_color is False
            assert loaded_config.verbose_output is True
        finally:
            os.unlink(config_file)
    
    def test_config_json_save_load(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            # Create config
            config = CLIConfig(timeout_seconds=600, parallel_execution=False)
            
            # Save to file
            config.save_to_file(config_file)
            
            # Load from file
            loaded_config = CLIConfig.load_from_file(config_file)
            
            assert loaded_config.timeout_seconds == 600
            assert loaded_config.parallel_execution is False
        finally:
            os.unlink(config_file)
    
    def test_config_manager(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            
            # Mock the default config path
            with patch.object(CLIConfig, 'get_default_config_path', return_value=config_path):
                manager = CLIConfigManager()
                
                # Update config
                manager.update_config(use_color=False, timeout_seconds=600)
                
                assert manager.config.use_color is False
                assert manager.config.timeout_seconds == 600
                
                # Create new manager (should load saved config)
                manager2 = CLIConfigManager()
                assert manager2.config.use_color is False
                assert manager2.config.timeout_seconds == 600
    
    def test_config_manager_reset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            
            with patch.object(CLIConfig, 'get_default_config_path', return_value=config_path):
                manager = CLIConfigManager()
                
                # Update config
                manager.update_config(use_color=False)
                assert manager.config.use_color is False
                
                # Reset to defaults
                manager.reset_to_defaults()
                assert manager.config.use_color is True  # Default value


class TestCLIIntegration:
    @pytest.mark.asyncio
    @patch('src.cli.MultiAgentCLI.setup_environment')
    @patch('src.cli.MultiAgentCLI.run_query')
    @patch('src.cli.MultiAgentCLI.display_results')
    async def test_main_with_query(self, mock_display, mock_run_query, mock_setup):
        mock_setup.return_value = True
        mock_run_query.return_value = {"final_output": {"summary": "Test"}}
        
        # Mock sys.argv
        with patch('sys.argv', ['cli.py', '-q', 'test query']):
            with patch('sys.exit') as mock_exit:
                try:
                    await main()
                except SystemExit:
                    pass
        
        mock_setup.assert_called_once()
        mock_run_query.assert_called_once()
        mock_display.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.cli.MultiAgentCLI.setup_environment')
    @patch('src.cli.MultiAgentCLI.setup_command')
    async def test_main_with_setup(self, mock_setup_command, mock_setup_env):
        mock_setup_env.return_value = True
        mock_setup_command.return_value = True
        
        # Mock sys.argv
        with patch('sys.argv', ['cli.py', '--setup']):
            with patch('sys.exit') as mock_exit:
                try:
                    await main()
                except SystemExit:
                    pass
        
        mock_setup_env.assert_called_once()
        mock_setup_command.assert_called_once()
    
    def test_argument_parsing(self):
        cli = MultiAgentCLI()
        parser = cli.create_parser()
        
        # Test various argument combinations
        args = parser.parse_args(["-q", "test", "--format", "json"])
        assert args.query == "test"
        assert args.format == "json"
        
        args = parser.parse_args(["--interactive", "--debug"])
        assert args.interactive is True
        assert args.debug is True
        
        args = parser.parse_args(["--setup", "--verbose", "--verbose"])
        assert args.setup is True
        assert args.verbose == 2
    
    def test_config_file_loading(self):
        cli = MultiAgentCLI()
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            cli._load_config_file("non_existent.yaml")
        
        # Test with invalid file format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            config_file = f.name
        
        try:
            with pytest.raises(ValueError):
                cli._load_config_file(config_file)
        finally:
            os.unlink(config_file)


class TestCLIErrorHandling:
    def test_display_error_results(self):
        cli = MultiAgentCLI()
        cli.console = Mock()
        
        result = {"error": "Test error message"}
        args = Mock()
        
        cli.display_results(result, args)
        
        # Should display error panel without saving results
        assert cli.console.print.called
    
    @pytest.mark.asyncio
    async def test_query_execution_error(self):
        cli = MultiAgentCLI()
        cli.workflow = Mock()
        
        # Create a proper async generator that raises an exception
        async def error_generator():
            raise Exception("Test error")
            yield  # This line will never be reached but makes it a generator
        
        cli.workflow.run_workflow_streaming = Mock(return_value=error_generator())
        
        # Mock Rich Progress context manager
        mock_progress = Mock()
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        mock_progress.add_task = Mock(return_value="task1")
        mock_progress.update = Mock()
        
        with patch('src.cli.Progress', return_value=mock_progress):
            with patch.object(cli, 'console') as mock_console:
                args = Mock()
                args.stream = True
                args.no_stream = False
                
                result = await cli.run_query("test query", args)
        
        assert "error" in result
        assert "Test error" in result["error"]
    
    def test_visualization_error(self):
        cli = MultiAgentCLI()
        cli.console = Mock()
        cli.workflow = Mock()
        cli.workflow.get_graph_visualization = Mock(side_effect=Exception("Viz error"))
        
        args = Mock()
        args.graph_format = "mermaid"
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(cli._show_visualization(args))
        finally:
            loop.close()
        
        # Should display error message
        assert cli.console.print.called


if __name__ == "__main__":
    pytest.main([__file__])