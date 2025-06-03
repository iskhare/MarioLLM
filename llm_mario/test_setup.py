#!/usr/bin/env python3

import sys
import os

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import gym_super_mario_bros
        print("✓ gym_super_mario_bros imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import gym_super_mario_bros: {e}")
        return False
    
    try:
        import anthropic
        print("✓ anthropic imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import anthropic: {e}")
        return False
    
    try:
        from agent import MarioEmulator, LLMAgent
        print("✓ Local agent modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import local modules: {e}")
        return False
    
    return True

def test_environment():
    """Test that the Mario environment can be created"""
    print("\nTesting environment creation...")
    
    try:
        from agent import MarioEmulator
        emulator = MarioEmulator(render_mode='rgb_array')
        print("✓ Mario emulator created successfully")
        
        # Test reset
        state = emulator.reset()
        print("✓ Environment reset successfully")
        
        # Test screenshot
        screenshot = emulator.get_screenshot()
        if screenshot:
            print("✓ Screenshot captured successfully")
        else:
            print("✗ Failed to capture screenshot")
            return False
        
        # Test game state
        game_state = emulator.get_game_state()
        print(f"✓ Game state retrieved: {game_state}")
        
        # Test step
        _, reward, done, truncated, info = emulator.step(1)  # Move right
        print(f"✓ Environment step executed, reward: {reward}")
        
        emulator.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_api_key():
    """Test API key configuration"""
    print("\nTesting API configuration...")
    
    import config
    if config.ANTHROPIC_API_KEY:
        print("✓ ANTHROPIC_API_KEY is set")
        return True
    else:
        print("✗ ANTHROPIC_API_KEY is not set")
        print("  Set it with: export ANTHROPIC_API_KEY=your_key_here")
        return False

def test_llm_agent():
    """Test LLM agent creation (without API call)"""
    print("\nTesting LLM agent creation...")
    
    try:
        import config
        if not config.ANTHROPIC_API_KEY:
            print("⚠ Skipping LLM agent test (no API key)")
            return True
            
        from agent import LLMAgent
        agent = LLMAgent()
        print("✓ LLM agent created successfully")
        
        # Test system prompt
        prompt = agent.get_system_prompt()
        if len(prompt) > 100:
            print("✓ System prompt generated")
        else:
            print("✗ System prompt seems too short")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ LLM agent test failed: {e}")
        return False

def main():
    print("=== LLM Mario Setup Test ===\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Test", test_environment),
        ("API Key Test", test_api_key),
        ("LLM Agent Test", test_llm_agent),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=== Test Summary ===")
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The setup is ready to use.")
        print("Run 'python main.py' to start playing Mario with LLM!")
    else:
        print("\n❌ Some tests failed. Please fix the issues before running the main script.")

if __name__ == "__main__":
    main() 