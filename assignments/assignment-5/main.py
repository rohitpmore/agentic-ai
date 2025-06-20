#!/usr/bin/env python3
"""
Main CLI interface for the AI Travel Agent & Expense Planner

This script provides a command-line interface to interact with the travel planning system.
"""

import sys
import os
import argparse
from typing import Dict, Any
import json
import logging
from travel_agent_system.core.langgraph_workflow import LangGraphTravelWorkflow
from travel_agent_system.utils.formatters import display_trip_summary
from config import config

# Add the current directory to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules with error handling
try:
    from travel_agent_system.core.langgraph_workflow import LangGraphTravelWorkflow
    from travel_agent_system.utils.formatters import display_trip_summary
except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules may not be available. Running in limited mode.")
    LangGraphTravelWorkflow = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_api_keys():
    """Check for missing API keys and exit if any are not found."""
    missing_keys = []
    if not config.gemini_api_key:
        missing_keys.append("GEMINI_API_KEY")
    if not config.openweather_api_key:
        missing_keys.append("OPENWEATHER_API_KEY")
    if not config.foursquare_api_key:
        missing_keys.append("FOURSQUARE_API_KEY")
    if not config.exchangerate_api_key:
        missing_keys.append("EXCHANGERATE_API_KEY")

    if missing_keys:
        logger.error("🛑 API Key Error: The following required API keys are missing:")
        for key in missing_keys:
            logger.error(f"  - {key}")
        logger.error("Please add them to your .env file and try again.")
        sys.exit(1)

def print_banner():
    """Print the application banner."""
    print("=" * 70)
    print("✈️  AI Travel Agent & Expense Planner")
    print("=" * 70)
    print("🌤️  Weather Agent: Climate analysis and recommendations")
    print("🎯 Attraction Agent: Points of interest discovery")
    print("🏨 Hotel Agent: Accommodation suggestions and pricing")
    print("📋 Itinerary Agent: Day-by-day trip planning with cost optimization")
    print("=" * 70)


def print_result(result: Dict[str, Any]):
    """
    Print a formatted travel plan result.
    
    Args:
        result: Travel plan result dictionary
    """
    print("\n" + "=" * 60)
    
    if result.get("status") == "error":
        print(f"❌ Status: ERROR")
        print(f"🏙️  Destination: {result.get('destination', 'Unknown')}")
        print(f"❌ Errors: {', '.join(result.get('errors', []))}")
        
        # Show any partial data
        partial_data = result.get("partial_data", {})
        if any(partial_data.values()):
            print("\n📊 Partial Data Available:")
            for source, data in partial_data.items():
                if data:
                    print(f"   ✅ {source.title()}: Data collected")
                else:
                    print(f"   ❌ {source.title()}: Failed")
    else:
        print(f"✅ Status: SUCCESS")
        print(f"🏙️  Destination: {result.get('destination', 'Unknown')}")
        
        # Trip Summary
        trip_summary = result.get("trip_summary", {})
        if trip_summary:
            overview = trip_summary.get("overview", {})
            if overview:
                print(f"\n📊 Trip Overview:")
                print(f"   📅 Total Days: {overview.get('total_days', 0)}")
                print(f"   💰 Total Cost: {overview.get('currency', 'USD')} {overview.get('total_cost', 0):.2f}")
                print(f"   📈 Daily Average: {overview.get('currency', 'USD')} {overview.get('daily_average', 0):.2f}")
            
            # Cost Breakdown
            cost_breakdown = trip_summary.get("cost_breakdown", {})
            if cost_breakdown.get("categories"):
                print(f"\n💰 Cost Breakdown:")
                for category, details in cost_breakdown["categories"].items():
                    print(f"   {category.title()}: {details['formatted']} ({details['percentage']}%)")
            
            # Highlights
            highlights = trip_summary.get("highlights", [])
            if highlights:
                print(f"\n⭐ Trip Highlights:")
                for highlight in highlights[:3]:
                    print(f"   • {highlight}")
            
            # Executive Summary
            exec_summary = trip_summary.get("executive_summary")
            if exec_summary:
                print(f"\n📝 Executive Summary:")
                print(f"   {exec_summary}")
    
    # Processing Summary
    processing_summary = result.get("processing_summary", {})
    if processing_summary:
        print(f"\n🔄 Processing Summary:")
        print(f"   Completed: {processing_summary.get('completed', 0)}/{processing_summary.get('total', 0)} agents")
        print(f"   Duration: {processing_summary.get('total_time', 0):.2f}s")
        
        if processing_summary.get("failed"):
            print(f"   Failed: {', '.join(processing_summary['failed'])}")
    
    print("=" * 60)


def interactive_mode(workflow: LangGraphTravelWorkflow):
    """
    Run the system in interactive mode.
    
    Args:
        workflow: The travel planner workflow instance
    """
    print("\n🔄 Interactive Mode - Type 'quit', 'exit', or 'q' to stop")
    print("💡 Try questions like:")
    print("   • 'Plan a 3-day trip to Paris'")
    print("   • 'Trip to Tokyo for 5 days with $2000 budget'")
    print("   • 'Visit London for a weekend'")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n🗺️  Your travel request: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("✈️ Safe travels!")
                break
            
            if not question:
                print("❌ Please enter a valid travel request.")
                continue
            
            print("\n🤖 Planning your trip...")
            result = workflow.query(question)
            print_result(result)
            
        except KeyboardInterrupt:
            print("\n✈️ Safe travels!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def single_query_mode(workflow: LangGraphTravelWorkflow, question: str):
    """
    Process a single travel query and exit.
    
    Args:
        workflow: The travel planner workflow instance
        question: The travel request to process
    """
    print(f"\n🤖 Planning trip: {question}")
    result = workflow.query(question)
    print_result(result)


def test_mode(workflow: LangGraphTravelWorkflow):
    """
    Run the system test suite.
    
    Args:
        workflow: The travel planner workflow instance
    """
    print("\n🧪 Running System Tests...")
    
    # Test API connections first
    print("\n1️⃣ Testing API Connections:")
    try:
        from travel_agent_system.utils.api_clients import APIClientManager
        api_manager = APIClientManager()
        
        # Test each API
        apis = ["weather", "foursquare", "currency"]
        all_connected = True
        
        for api in apis:
            try:
                if api == "weather":
                    result = api_manager.get_current_weather("London")
                elif api == "foursquare":
                    result = api_manager.search_places("Paris", "attractions", limit=1)
                elif api == "currency":
                    result = api_manager.get_exchange_rates("USD")
                
                print(f"   ✅ {api.title()} API: Connected")
            except Exception as e:
                print(f"   ❌ {api.title()} API: Failed ({str(e)[:50]}...)")
                all_connected = False
        
        if not all_connected:
            print("\n⚠️ Some APIs failed. Tests will use fallback data.")
    
    except Exception as e:
        print(f"   ❌ API testing failed: {e}")
    
    print("\n2️⃣ Running End-to-End Tests:")
    
    # Test scenarios
    test_cases = [
        "Trip to Paris for 3 days",
        "Visit Tokyo with $1000 budget",
        "Weekend in London"
    ]
    
    results = {}
    for test_case in test_cases:
        try:
            print(f"   🧪 Testing: {test_case}")
            result = workflow.query(test_case)
            
            if result.get("status") == "success":
                results[test_case] = "✅ PASSED"
            else:
                results[test_case] = f"⚠️ PARTIAL ({len(result.get('errors', []))} errors)"
                
        except Exception as e:
            results[test_case] = f"❌ FAILED ({str(e)[:30]}...)"
    
    print(f"\n📊 Test Summary:")
    for test_case, result in results.items():
        print(f"   {result}: {test_case}")


def setup_mode():
    """
    Initialize the system components and test connections.
    """
    print("\n⚙️ Setting up AI Travel Agent System...")
    
    print("🔧 Checking configuration...")
    try:
        from config import Config
        config = Config.from_env()
        print("✅ Configuration loaded")
        
        # Check API keys
        api_keys = [
            ("OpenWeather API", config.openweather_api_key),
            ("Foursquare API", config.foursquare_api_key),
            ("ExchangeRate API", config.exchangerate_api_key)
        ]
        
        missing_keys = []
        for name, key in api_keys:
            if key and key != "your_key_here":
                print(f"   ✅ {name}: Configured")
            else:
                print(f"   ❌ {name}: Missing")
                missing_keys.append(name)
        
        if missing_keys:
            print(f"\n⚠️ Missing API keys: {', '.join(missing_keys)}")
            print("💡 Please configure your .env file with:")
            for name, _ in api_keys:
                if name in missing_keys:
                    env_var = name.split()[0].upper() + "_API_KEY"
                    print(f"   {env_var}=your_key_here")
            return False
        
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        print("💡 Create a .env file with required API keys")
        return False
    
    print("\n🔌 Testing system initialization...")
    try:
        if LangGraphTravelWorkflow is None:
            print("❌ Cannot test - import failed")
            return False
            
        workflow = LangGraphTravelWorkflow()
        
        # Quick connection test
        test_result = workflow.query("Test connection to London")
        
        if test_result.get("status") in ["success", "error"]:
            print("✅ System ready!")
            return True
        else:
            print("❌ System initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Travel Agent & Expense Planner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                               # Interactive mode
  python main.py -q "Trip to Paris for 3 days" # Single query
  python main.py --test                        # Run tests
  python main.py --setup                       # Initialize system
        """
    )
    
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Process a single travel request and exit"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the system test suite"
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Initialize system components and test connections"
    )
    
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip the banner display"
    )
    
    parser.add_argument(
        "--save-graph",
        type=str,
        metavar="FILENAME",
        help="Save workflow graph visualization as PNG"
    )
    
    args = parser.parse_args()
    
    # Always check for API keys first
    check_api_keys()

    # Print banner unless suppressed
    if not args.no_banner:
        print_banner()
    
    try:
        # Handle different modes
        if args.setup:
            setup_mode()
        elif args.test:
            # Initialize the workflow for testing
            if LangGraphTravelWorkflow is None:
                print("❌ Cannot run tests - system import failed")
                sys.exit(1)
            print("\n🚀 Initializing Travel Agent System...")
            workflow = LangGraphTravelWorkflow()
            print("✅ System initialized successfully!")
            test_mode(workflow)
        elif args.save_graph:
            # Save workflow graph visualization
            if LangGraphTravelWorkflow is None:
                print("❌ Cannot generate graph - system import failed")
                sys.exit(1)
            print(f"\n📊 Generating workflow graph: {args.save_graph}")
            workflow = LangGraphTravelWorkflow()
            workflow.save_graph_image(args.save_graph)
            print(f"✅ Graph saved as {args.save_graph}")
        elif args.query:
            # Initialize the workflow for single query
            if LangGraphTravelWorkflow is None:
                print("❌ Cannot process query - system import failed")
                sys.exit(1)
            print("\n🚀 Initializing Travel Agent System...")
            workflow = LangGraphTravelWorkflow()
            print("✅ System initialized successfully!")
            single_query_mode(workflow, args.query)
        else:
            # Initialize the workflow for interactive mode
            if LangGraphTravelWorkflow is None:
                print("❌ Cannot start interactive mode - system import failed")
                print("💡 Run 'python main.py --setup' to diagnose issues")
                sys.exit(1)
            print("\n🚀 Initializing Travel Agent System...")
            workflow = LangGraphTravelWorkflow()
            print("✅ System initialized successfully!")
            interactive_mode(workflow)
            
    except KeyboardInterrupt:
        print("\n✈️ Safe travels!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        print("💡 Make sure your .env file contains valid API keys:")
        print("   • OPENWEATHER_API_KEY")
        print("   • FOURSQUARE_API_KEY") 
        print("   • EXCHANGERATE_API_KEY")
        print("   • GEMINI_API_KEY (optional)")
        print("💡 Run 'python main.py --setup' to test your configuration")
        sys.exit(1)


if __name__ == "__main__":
    main()