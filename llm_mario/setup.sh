#!/bin/bash

set -e

echo "🍄 LLM Mario Setup Script 🍄"
echo "================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8 or higher is required. Current version: $python_version"
    exit 1
fi
echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if API key is set
echo "Checking API key configuration..."
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  ANTHROPIC_API_KEY is not set"
    echo "   Set it with: export ANTHROPIC_API_KEY=your_key_here"
    echo "   Or add it to your ~/.bashrc or ~/.zshrc"
else
    echo "✅ ANTHROPIC_API_KEY is configured"
fi

# Run setup test
echo "Running setup tests..."
python test_setup.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Quick start:"
echo "  1. Set API key: export ANTHROPIC_API_KEY=your_key"
echo "  2. Activate environment: source venv/bin/activate"
echo "  3. Run LLM agent: python main.py"
echo "  4. Run manual demo: python demo_manual.py"
echo ""
echo "For more options, see README.md" 