"""
Output formatters for travel planning results - Placeholder for Stage 5 implementation
"""

from typing import Dict, Any


class TripFormatter:
    """
    Utility for formatting travel planning output.
    Will be implemented in Stage 5.
    """
    
    def __init__(self):
        """Initialize formatter"""
        pass
    
    def format_trip_summary(self, trip_data: Dict[str, Any]) -> str:
        """Format complete trip summary - placeholder"""
        return "Trip summary formatting - to be implemented in Stage 5"
    
    def format_cost_breakdown(self, costs: Dict[str, float]) -> str:
        """Format cost breakdown - placeholder"""
        return "Cost breakdown formatting - to be implemented in Stage 5"
    
    def format_itinerary(self, itinerary: Dict[str, Any]) -> str:
        """Format day-by-day itinerary - placeholder"""
        return "Itinerary formatting - to be implemented in Stage 5"

def display_trip_summary(trip_data: Dict[str, Any]):
    """
    Displays the final trip summary in a user-friendly format.
    """
    if not trip_data or trip_data.get("status") != "SUCCESS":
        print("\nCould not generate a trip plan based on the provided information.")
        if trip_data.get("error"):
            print(f"Error: {trip_data['error']}")
        return

    print("\n============================================================")
    print(f"✅ Status: {trip_data['status']}")
    print(f"🏙️  Destination: {trip_data['destination']}")
    
    overview = trip_data.get('overview', {})
    if overview:
        print("\n📊 Trip Overview:")
        print(f"   📅 Total Days: {overview.get('total_days', 'N/A')}")
        print(f"   💰 Total Cost: {overview.get('total_cost_native_currency', 'N/A')} {overview.get('native_currency', 'USD')}")
        print(f"   📈 Daily Average: {overview.get('daily_average_native_currency', 'N/A')} {overview.get('native_currency', 'USD')}")

    cost_breakdown = trip_data.get('cost_breakdown', {})
    if cost_breakdown:
        print("\n💰 Cost Breakdown:")
        for category, details in cost_breakdown.items():
            print(f"   {category.capitalize()}: {details.get('cost_native', 'N/A')} {overview.get('native_currency', 'USD')} ({details.get('percentage', 'N/A')}%)")

    highlights = trip_data.get('highlights', [])
    if highlights:
        print("\n⭐ Trip Highlights:")
        for highlight in highlights:
            print(f"   • {highlight}")

    summary = trip_data.get('executive_summary', 'No summary available.')
    print(f"\n📝 Executive Summary:\n   {summary}")
    
    processing_summary = trip_data.get('processing_summary', {})
    if processing_summary:
        print("\n🔄 Processing Summary:")
        print(f"   Completed: {processing_summary.get('completed_agents', 0)}/{processing_summary.get('total_agents', 0)} agents")
        print(f"   Duration: {processing_summary.get('duration_seconds', 0):.2f}s")
    print("============================================================")