#!/usr/bin/env python3
"""
Test CLI output formatting for complete travel plans
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from travel_agent_system.core.langgraph_workflow import LangGraphTravelWorkflow
from travel_agent_system.core.graph_state import create_initial_state

def test_cli_formatted_output():
    """Test the complete CLI travel plan output"""
    print("🧪 Testing CLI formatted travel plan output...")
    
    # Create workflow
    workflow = LangGraphTravelWorkflow()
    
    # Create structured input state (bypass natural language parsing)
    initial_state = create_initial_state(
        destination="Paris",
        budget=1500,
        travel_dates={
            "start_date": "2025-08-01",
            "end_date": "2025-08-04"
        },
        currency="USD",
        preferences={"pace": "moderate"}
    )
    
    print(f"📍 Testing destination: {initial_state['destination']}")
    print(f"💰 Budget: ${initial_state['budget']}")
    
    # Execute workflow using the query method (like CLI does)
    print("\n🚀 Executing workflow via query method...")
    result = workflow._parse_query("")  # Dummy query
    initial_state_dict = dict(initial_state)
    
    final_state = None
    for state in workflow.graph.stream(initial_state, config={"configurable": {"thread_id": "test_paris"}}):
        final_state = state
    
    if final_state:
        last_node_state = list(final_state.values())[-1]
        formatted_result = workflow._format_results(last_node_state)
        
        print("\n" + "="*60)
        print("🌟 COMPLETE TRAVEL PLAN OUTPUT")
        print("="*60)
        
        print(f"✅ Success: {formatted_result.get('success')}")
        print(f"🏙️  Destination: {formatted_result.get('destination')}")
        
        # Show itinerary details
        itinerary = formatted_result.get('itinerary')
        if itinerary:
            print(f"\n📊 Trip Overview:")
            print(f"   📅 Total Days: {itinerary.get('total_days', 0)}")
            print(f"   💰 Total Cost: ${itinerary.get('total_cost', 0):.2f}")
            
            # Show cost breakdown
            cost_breakdown = itinerary.get('cost_breakdown', {})
            if cost_breakdown:
                print(f"\n💰 Cost Breakdown:")
                for category, amount in cost_breakdown.items():
                    percentage = (amount / itinerary.get('total_cost', 1)) * 100
                    print(f"   {category.title()}: ${amount:.2f} ({percentage:.1f}%)")
            
            # Show daily plans
            daily_plans = itinerary.get('daily_plans', [])
            print(f"\n📅 Daily Itinerary ({len(daily_plans)} days):")
            for i, day in enumerate(daily_plans[:2], 1):  # Show first 2 days
                activities = len(day.get('morning', []) + day.get('afternoon', []) + day.get('evening', []))
                print(f"   Day {i}: {activities} activities planned")
                if day.get('weather'):
                    temp = day['weather'].get('temperature', {}).get('avg', 'N/A')
                    desc = day['weather'].get('description', 'N/A')
                    print(f"      Weather: {temp}°C, {desc}")
            
            # Show recommendations
            recommendations = itinerary.get('recommendations', [])
            if recommendations:
                print(f"\n⭐ Recommendations:")
                for rec in recommendations[:3]:  # Show first 3
                    print(f"   • {rec}")
        
        # Show data sources
        data_sources = []
        if formatted_result.get('weather'):
            data_sources.append("Weather")
        if formatted_result.get('attractions'):
            data_sources.append("Attractions")
        if formatted_result.get('hotels'):
            data_sources.append("Hotels")
        
        print(f"\n📊 Data Sources: {', '.join(data_sources)}")
        print("="*60)
        
        return formatted_result
    
    return None

if __name__ == "__main__":
    test_cli_formatted_output()