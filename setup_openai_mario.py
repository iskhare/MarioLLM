import os
import subprocess
import sys

def setup_openai_mario():
    """Setup script for OpenAI Mario agent"""
    
    print("🎮 Setting up OpenAI Mario Agent...")
    
    # Install requirements
    print("📦 Installing requirements...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    # Check for .env file
    if not os.path.exists('.env'):
        print("\n⚠️  Creating .env file...")
        with open('.env', 'w') as f:
            f.write("# Add your OpenAI API key here\n")
            f.write("OPENAI_API_KEY=your_key_here\n")
        print("📝 Please edit .env file and add your OpenAI API key")
    
    # Check if API key is set
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your_key_here':
        print("\n🔑 Please set your OpenAI API key in the .env file")
        print("Get your API key from: https://platform.openai.com/api-keys")
        return False
    
    print("\n✅ Setup complete! You can now run:")
    print("python openai_mario_runner.py")
    return True

if __name__ == "__main__":
    setup_openai_mario() 