#!/usr/bin/env python3
"""
Test LangGraph workflow with structured input to bypass natural language parsing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from travel_agent_system.core.langgraph_workflow import LangGraphTravelWorkflow
from travel_agent_system.core.graph_state import create_initial_state

def test_structured_input():
    """Test workflow with structured input data"""
    print("🧪 Testing LangGraph workflow with structured input...")
    
    # Create workflow
    workflow = LangGraphTravelWorkflow()
    
    # Create structured input state (bypass natural language parsing)
    initial_state = create_initial_state(
        destination="London",
        budget=1000,
        travel_dates={
            "start_date": "2025-07-01",
            "end_date": "2025-07-03"
        },
        currency="USD",
        preferences={"pace": "moderate"}
    )
    
    print(f"📍 Testing destination: {initial_state['destination']}")
    print(f"💰 Budget: ${initial_state['budget']}")
    print(f"📅 Dates: {initial_state['travel_dates']['start_date']} to {initial_state['travel_dates']['end_date']}")
    
    # Execute workflow
    print("\n🚀 Executing LangGraph workflow...")
    
    final_state = None
    for state in workflow.graph.stream(initial_state, config={"configurable": {"thread_id": "test_london"}}):
        final_state = state
        print(f"📊 Step: {list(state.keys())}")
    
    # Get final result
    if final_state:
        last_node_state = list(final_state.values())[-1]
        result = workflow._format_results(last_node_state)
        
        print(f"\n✅ Workflow completed!")
        print(f"🏙️  Destination: {result.get('destination')}")
        print(f"🎯 Success: {result.get('success')}")
        
        # Check for itinerary
        itinerary = result.get('itinerary')
        if itinerary:
            print(f"📋 Itinerary generated: {len(itinerary.get('daily_plans', []))} days")
            print(f"💰 Total cost: ${itinerary.get('total_cost', 0):.2f}")
            
            # Show daily plans
            for i, day in enumerate(itinerary.get('daily_plans', [])[:2], 1):
                print(f"   Day {i}: {len(day.get('morning', []) + day.get('afternoon', []) + day.get('evening', []))} activities")
        else:
            print("❌ No itinerary generated")
            
        # Check for errors
        if result.get('errors'):
            print(f"⚠️  Errors: {result['errors']}")
            
        return result
    else:
        print("❌ Workflow execution failed")
        return None

if __name__ == "__main__":
    test_structured_input()