#!/usr/bin/env python3
"""
Main CLI interface for the Multi-Agent System

This script provides a command-line interface to interact with the multi-agent system.
"""

import sys
import argparse
from typing import Dict, Any

# Add the current directory to the Python path for imports
sys.path.append('.')

from multi_agent_system import MultiAgentWorkflow


def print_banner():
    """Print the application banner."""
    print("=" * 70)
    print("🤖 Multi-Agent System with Supervisor Node")
    print("=" * 70)
    print("🏛️  RAG Agent: USA Economy questions")
    print("🤖 LLM Agent: General Knowledge questions")
    print("🌐 Web Crawler Agent: Real-time/Current Events questions")
    print("✅ Validation Agent: Quality control with feedback loops")
    print("=" * 70)


def print_result(result: Dict[str, Any]):
    """
    Print a formatted query result.
    
    Args:
        result: Query result dictionary
    """
    print("\n" + "=" * 50)
    print(f"❓ Question: {result['question']}")
    print("-" * 50)
    print(f"📝 Answer: {result['answer']}")
    print("-" * 50)
    print(f"✅ Status: {result['status']}")
    print(f"🔍 Validation: {result['validation_status']}")
    if result['retry_count'] > 0:
        print(f"🔄 Retries: {result['retry_count']}")
    print("=" * 50)


def interactive_mode(workflow: MultiAgentWorkflow):
    """
    Run the system in interactive mode.
    
    Args:
        workflow: The multi-agent workflow instance
    """
    print("\n🔄 Interactive Mode - Type 'quit', 'exit', or 'q' to stop")
    print("💡 Try questions like:")
    print("   • 'What is the US GDP structure?'")
    print("   • 'Explain quantum computing'")
    print("   • 'Latest news about AI today'")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                print("❌ Please enter a valid question.")
                continue
            
            print("\n🤖 Processing your question...")
            result = workflow.query(question)
            print_result(result)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def single_query_mode(workflow: MultiAgentWorkflow, question: str):
    """
    Process a single query and exit.
    
    Args:
        workflow: The multi-agent workflow instance
        question: The question to process
    """
    print(f"\n🤖 Processing question: {question}")
    result = workflow.query(question)
    print_result(result)


def test_mode(workflow: MultiAgentWorkflow):
    """
    Run the system test suite.
    
    Args:
        workflow: The multi-agent workflow instance
    """
    print("\n🧪 Running System Tests...")
    
    # Test connections first
    print("\n1️⃣ Testing External Connections:")
    connections = workflow.test_connections()
    
    if not all(connections.values()):
        print("\n❌ Some connections failed. Please check your API keys and configuration.")
        return
    
    print("\n2️⃣ Running End-to-End Tests:")
    results = workflow.test_system()
    
    print(f"\n📊 Test Summary:")
    for query, result in results.items():
        if 'error' in result:
            print(f"❌ '{query}': {result['error']}")
        else:
            status_emoji = "✅" if result['status'] == 'success' else "⚠️"
            print(f"{status_emoji} '{query}': {result['status']} ({result['validation_status']})")


def setup_mode(workflow: MultiAgentWorkflow, force_recreate: bool = False):
    """
    Initialize the system components.
    
    Args:
        workflow: The multi-agent workflow instance
        force_recreate: Whether to force recreation of vector store
    """
    print("\n⚙️ Setting up Multi-Agent System...")
    
    print("📚 Initializing vector store...")
    try:
        workflow.initialize_vector_store(force_recreate=force_recreate)
        print("✅ Vector store ready")
    except Exception as e:
        print(f"❌ Vector store setup failed: {e}")
        return False
    
    print("🔌 Testing connections...")
    connections = workflow.test_connections()
    
    if all(connections.values()):
        print("✅ All systems ready!")
        return True
    else:
        print("❌ Some connections failed. Please check your configuration.")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent System with Supervisor Node",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive mode
  python main.py -q "What is US GDP?"     # Single query
  python main.py --test                   # Run tests
  python main.py --setup                  # Initialize system
        """
    )
    
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Process a single query and exit"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the system test suite"
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Initialize system components (vector store, etc.)"
    )
    
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation of vector store (use with --setup)"
    )
    
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip the banner display"
    )
    
    args = parser.parse_args()
    
    # Print banner unless suppressed
    if not args.no_banner:
        print_banner()
    
    try:
        # Initialize the workflow
        print("\n🚀 Initializing Multi-Agent System...")
        workflow = MultiAgentWorkflow()
        print("✅ System initialized successfully!")
        
        # Handle different modes
        if args.setup:
            setup_mode(workflow, args.force_recreate)
        elif args.test:
            test_mode(workflow)
        elif args.query:
            single_query_mode(workflow, args.query)
        else:
            interactive_mode(workflow)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        print("💡 Make sure your .env file contains valid API keys:")
        print("   • GEMINI_API_KEY")
        print("   • TAVILY_API_KEY")
        sys.exit(1)


if __name__ == "__main__":
    main() 