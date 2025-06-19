#!/usr/bin/env python3
"""
Connection testing utility for Stage 2 verification
"""

import sys
import os

# Add the current directory to the Python path for imports
sys.path.append('.')

try:
    from travel_agent_system.utils.api_clients import APIClient
    from config import config
    
    def test_api_connections():
        """Test all API connections"""
        print("🔌 Testing API Connections...")
        print("=" * 50)
        
        # Check if API keys are configured
        print("📋 Checking API Key Configuration:")
        try:
            print(f"✅ GEMINI_API_KEY: {'Set' if config.gemini_api_key else 'Missing'}")
            print(f"✅ OPENWEATHER_API_KEY: {'Set' if config.openweather_api_key else 'Missing'}")
            print(f"✅ FOURSQUARE_API_KEY: {'Set' if config.foursquare_api_key else 'Missing'}")
            print(f"✅ EXCHANGERATE_API_KEY: {'Set' if config.exchangerate_api_key else 'Missing'}")
        except Exception as e:
            print(f"❌ Configuration Error: {e}")
            print("💡 Please create .env file with required API keys")
            return False
        
        print("\n🌐 Testing API Connections:")
        
        try:
            client = APIClient()
            results = client.test_all_connections()
            
            for service, success in results.items():
                status = "✅" if success else "❌"
                print(f"{status} {service.title()} API: {'Connected' if success else 'Failed'}")
            
            all_connected = all(results.values())
            print(f"\n🎯 Overall Status: {'All APIs Connected' if all_connected else 'Some APIs Failed'}")
            return all_connected
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    if __name__ == "__main__":
        success = test_api_connections()
        sys.exit(0 if success else 1)
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure you have created .env file and installed dependencies")
    print("💡 Run: pip install -r requirements.txt")
    sys.exit(1)