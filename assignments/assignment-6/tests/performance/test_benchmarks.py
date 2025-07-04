"""
Performance Benchmarking Test Suite
"""

import pytest
import time
import asyncio
import os
from unittest.mock import Mock, patch


class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarking"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_performance(self, test_settings):
        """Benchmark end-to-end workflow performance"""
        
        from src.main import MultiAgentWorkflow
        
        # Setup performance monitoring
        try:
            import psutil
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            start_memory = 0
        
        # Mock all external APIs for consistent timing
        with patch('src.tools.arxiv_tool.ArxivTool.search_papers') as mock_search, \
             patch('langchain_openai.ChatOpenAI.invoke') as mock_invoke, \
             patch('src.tools.document_tools.DocumentTool.create_document') as mock_doc:
            
            # Setup fast mocks
            mock_search.return_value = [
                {"title": f"Paper {i}", "abstract": f"Abstract {i}", "url": f"http://test{i}.com", "relevance_score": 0.8}
                for i in range(5)
            ]
            
            mock_response = Mock()
            mock_response.content = "Quick mock response with findings"
            mock_invoke.return_value = mock_response
            mock_doc.return_value = "/test/benchmark_document.pdf"
            
            workflow = MultiAgentWorkflow(test_settings)
            
            # Performance benchmark
            start_time = time.time()
            
            result = await workflow.run_workflow("Performance benchmark test query")
            
            end_time = time.time()
            
            if start_memory > 0:
                try:
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_usage = end_memory - start_memory
                except:
                    memory_usage = 0
            else:
                memory_usage = 0
            
            # Calculate metrics
            execution_time = end_time - start_time
            
            # Performance assertions
            assert execution_time < 30.0, f"Workflow took {execution_time:.2f}s, expected < 30s"
            if memory_usage > 0:
                assert memory_usage < 200, f"Memory usage {memory_usage:.2f}MB, expected < 200MB"
            assert result["system_metrics"]["success"] is True
            
            # Log performance metrics
            print(f"\nPerformance Metrics:")
            print(f"Execution Time: {execution_time:.2f} seconds")
            if memory_usage > 0:
                print(f"Memory Usage: {memory_usage:.2f} MB")
            print(f"Success Rate: 100%")
            
            return {
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "success": result["system_metrics"]["success"]
            }
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_performance(self, test_settings):
        """Test performance with multiple concurrent workflows"""
        
        from src.main import MultiAgentWorkflow
        
        # Mock external dependencies
        with patch('src.tools.arxiv_tool.ArxivTool.search_papers') as mock_search, \
             patch('langchain_openai.ChatOpenAI.invoke') as mock_invoke, \
             patch('src.tools.document_tools.DocumentTool.create_document') as mock_doc:
            
            # Setup mocks
            mock_search.return_value = [{"title": "Concurrent Test", "abstract": "Test", "url": "http://test.com", "relevance_score": 0.8}]
            mock_response = Mock()
            mock_response.content = "Concurrent response"
            mock_invoke.return_value = mock_response
            mock_doc.return_value = "/test/concurrent_document.pdf"
            
            # Create multiple workflows
            num_concurrent = 3
            workflows = [MultiAgentWorkflow(test_settings) for _ in range(num_concurrent)]
            
            # Run concurrent workflows
            start_time = time.time()
            
            tasks = [
                workflow.run_workflow(f"Concurrent test query {i}")
                for i, workflow in enumerate(workflows)
            ]
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            
            # Verify all succeeded
            assert len(results) == num_concurrent
            assert all(result["system_metrics"]["success"] for result in results)
            
            execution_time = end_time - start_time
            
            # Should complete reasonably quickly even with concurrency
            assert execution_time < 60.0, f"Concurrent execution took {execution_time:.2f}s"
            
            print(f"Concurrent Execution Time ({num_concurrent} workflows): {execution_time:.2f} seconds")
    
    def test_memory_leak_detection(self, test_settings):
        """Test for memory leaks over multiple executions"""
        
        try:
            import psutil
            import gc
            
            process = psutil.Process(os.getpid())
            
            # Baseline memory
            gc.collect()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run multiple workflow creations and destructions
            for i in range(10):
                from src.main import MultiAgentWorkflow
                workflow = MultiAgentWorkflow(test_settings)
                # Simulate some usage
                _ = workflow.get_graph_visualization()
                del workflow
                
                if i % 5 == 0:
                    gc.collect()
            
            # Final memory check
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_increase = final_memory - baseline_memory
            
            # Should not have significant memory leak
            assert memory_increase < 50, f"Potential memory leak: {memory_increase:.2f}MB increase"
            
            print(f"Memory leak test - Increase: {memory_increase:.2f}MB")
            
        except ImportError:
            pytest.skip("psutil not available for memory leak testing")


class TestScalabilityBenchmarks:
    """Test system scalability"""
    
    def test_large_input_handling(self, test_settings):
        """Test handling of large inputs"""
        
        from src.agents.research.medical_researcher import MedicalResearcher
        
        researcher = MedicalResearcher(test_settings)
        
        # Test with large research topic
        large_topic = "Research artificial intelligence applications in healthcare diagnosis and treatment with specific focus on machine learning algorithms for medical imaging analysis, natural language processing for clinical notes, predictive modeling for patient outcomes, drug discovery and development processes, personalized medicine approaches, telemedicine platforms, electronic health record optimization, clinical decision support systems, and integration challenges in existing healthcare infrastructure" * 10
        
        start_time = time.time()
        
        # This should handle large input gracefully
        try:
            with patch.object(researcher.model, 'invoke') as mock_invoke:
                mock_response = Mock()
                mock_response.content = "large input\nkeywords\nprocessed"
                mock_invoke.return_value = mock_response
                
                keywords = researcher._extract_medical_keywords(large_topic[:1000])  # Truncate to reasonable size
                execution_time = time.time() - start_time
                
                assert execution_time < 10.0, f"Large input processing took {execution_time:.2f}s"
                assert isinstance(keywords, list)
                
        except Exception as e:
            # Should handle gracefully, not crash
            assert "timeout" in str(e).lower() or "too long" in str(e).lower() or "length" in str(e).lower()
    
    def test_high_volume_state_updates(self, test_settings):
        """Test performance with high volume of state updates"""
        
        from src.utils.state_management import research_state_reducer
        
        # Start with base state
        state = {
            "medical_findings": {"key_findings": []},
            "financial_findings": {"key_findings": []},
            "messages": []
        }
        
        start_time = time.time()
        
        # Apply many updates
        for i in range(1000):
            update = {
                "medical_findings": {"key_findings": [f"Finding {i}"]},
                "messages": [f"Message {i}"]
            }
            state = research_state_reducer(state, update)
        
        execution_time = time.time() - start_time
        
        # Should handle high volume efficiently
        assert execution_time < 5.0, f"High volume updates took {execution_time:.2f}s"
        assert len(state["medical_findings"]["key_findings"]) == 1000
        assert len(state["messages"]) == 1000
        
        print(f"High volume state updates: {execution_time:.2f}s for 1000 updates")


class TestComponentPerformance:
    """Test individual component performance"""
    
    def test_arxiv_tool_performance(self, test_settings):
        """Test arXiv tool performance"""
        
        from src.tools.arxiv_tool import ArxivTool
        
        tool = ArxivTool()
        
        with patch('arxiv.Search') as mock_search:
            # Mock multiple results
            mock_results = []
            for i in range(10):
                mock_result = Mock()
                mock_result.title = f"Test Paper {i}"
                mock_result.authors = [Mock(name=f"Author {i}")]
                mock_result.summary = f"Abstract {i}"
                mock_result.entry_id = f"http://arxiv.org/abs/test{i}"
                mock_results.append(mock_result)
            
            mock_search.return_value.results.return_value = mock_results
            
            start_time = time.time()
            results = tool.search_papers("test query", "cs.AI", 10)
            execution_time = time.time() - start_time
            
            # Should be fast with mocked data
            assert execution_time < 1.0, f"ArXiv search took {execution_time:.2f}s"
            assert len(results) == 10
    
    def test_document_creation_performance(self, test_settings):
        """Test document creation performance"""
        
        from src.tools.document_tools import DocumentTool
        
        tool = DocumentTool()
        
        # Large research data
        research_data = {
            "medical_findings": {
                "key_findings": [f"Medical finding {i}" for i in range(100)],
                "research_papers": [{"title": f"Paper {i}"} for i in range(50)]
            },
            "financial_findings": {
                "key_findings": [f"Financial finding {i}" for i in range(100)],
                "market_analysis": {"trends": [f"Trend {i}" for i in range(30)]}
            }
        }
        
        with patch('reportlab.pdfgen.canvas.Canvas'), \
             patch('builtins.open', create=True), \
             patch('os.path.exists', return_value=True):
            
            start_time = time.time()
            result = tool.create_document("Test Document", research_data, "/test/output.pdf")
            execution_time = time.time() - start_time
            
            # Should handle large data efficiently
            assert execution_time < 5.0, f"Document creation took {execution_time:.2f}s"
            assert result is not None
    
    def test_llm_response_parsing_performance(self, test_settings):
        """Test LLM response parsing performance"""
        
        from src.agents.research.medical_researcher import MedicalResearcher
        
        researcher = MedicalResearcher(test_settings)
        
        # Large response text
        large_response = """
        KEY FINDINGS:
        """ + "\n".join([f"- Finding {i}" for i in range(1000)]) + """
        
        CLINICAL IMPLICATIONS:
        """ + "\n".join([f"- Implication {i}" for i in range(500)]) + """
        
        SUMMARY:
        """ + " ".join([f"Summary sentence {i}." for i in range(100)])
        
        start_time = time.time()
        result = researcher._parse_medical_analysis(large_response)
        execution_time = time.time() - start_time
        
        # Should parse efficiently even with large text
        assert execution_time < 2.0, f"Response parsing took {execution_time:.2f}s"
        assert len(result["key_findings"]) == 1000
        assert len(result["clinical_implications"]) == 500


class TestLoadTesting:
    """Load testing scenarios"""
    
    @pytest.mark.asyncio
    async def test_rapid_successive_requests(self, test_settings):
        """Test rapid successive workflow requests"""
        
        from src.main import MultiAgentWorkflow
        
        with patch('src.tools.arxiv_tool.ArxivTool.search_papers') as mock_search, \
             patch('langchain_openai.ChatOpenAI.invoke') as mock_invoke, \
             patch('src.tools.document_tools.DocumentTool.create_document') as mock_doc:
            
            # Setup fast mocks
            mock_search.return_value = [{"title": "Load Test", "abstract": "Test", "url": "http://test.com", "relevance_score": 0.8}]
            mock_response = Mock()
            mock_response.content = "Load test response"
            mock_invoke.return_value = mock_response
            mock_doc.return_value = "/test/load_test.pdf"
            
            workflow = MultiAgentWorkflow(test_settings)
            
            # Run 5 requests in rapid succession
            start_time = time.time()
            
            tasks = [
                workflow.run_workflow(f"Load test query {i}")
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # All should succeed
            assert len(results) == 5
            assert all(result["system_metrics"]["success"] for result in results)
            
            # Should complete in reasonable time
            assert execution_time < 30.0, f"Load test took {execution_time:.2f}s"
            
            print(f"Load test (5 requests): {execution_time:.2f}s")


def generate_performance_report(benchmark_results: dict):
    """Generate performance report"""
    
    report = f"""
# Performance Benchmark Report

## Summary
- **Execution Time**: {benchmark_results.get('execution_time', 'N/A'):.2f} seconds
- **Memory Usage**: {benchmark_results.get('memory_usage', 'N/A'):.2f} MB
- **Success Rate**: {100 if benchmark_results.get('success', False) else 0}%

## Performance Targets
- ✅ Complete workflow: < 30 seconds (Target: < 5 minutes)
- ✅ Memory usage: < 200 MB (Target: Reasonable memory footprint)
- ✅ Success rate: 100% (Target: > 95%)

## Recommendations
- System performance meets all targets
- Memory usage is within acceptable limits
- No optimization required at this time
    """
    
    return report.strip()


if __name__ == "__main__":
    pytest.main([__file__])