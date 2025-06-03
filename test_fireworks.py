#!/usr/bin/env python3
"""
Test script to verify Fireworks API integration
"""

import os
from fireworks.client import Fireworks

def test_fireworks_api():
    """Test basic Fireworks API functionality"""
    
    # Check if API key is set
    api_key = os.getenv('FIREWORKS_API_KEY')
    if not api_key:
        print("❌ FIREWORKS_API_KEY environment variable not set")
        print("Please set it with: export FIREWORKS_API_KEY='your_api_key_here'")
        return False
    
    try:
        # Initialize client
        client = Fireworks(api_key=api_key)
        print("✅ Fireworks client initialized successfully")
        
        # Test a simple API call
        response = client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Respond in JSON format."},
                {"role": "user", "content": "Say hello and provide a test response in JSON format with keys 'message' and 'status'."}
            ],
            max_tokens=100
        )
        
        print("✅ API call successful!")
        print(f"Response: {response.choices[0].message.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Fireworks API: {e}")
        return False

if __name__ == "__main__":
    print("Testing Fireworks API integration...")
    success = test_fireworks_api()
    
    if success:
        print("\n🎉 Fireworks API integration test passed!")
        print("You can now run the Mario LLM agent with Fireworks API.")
    else:
        print("\n💥 Fireworks API integration test failed!")
        print("Please check your API key and network connection.") 