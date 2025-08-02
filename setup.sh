#!/bin/bash

# Transformers Examples Setup Script
echo "🚀 Setting up Transformers Examples repository..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment (optional but recommended)
if [ "$1" == "--venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "📝 Please edit .env file and add your Hugging Face token"
    echo "   You can get a token from: https://huggingface.co/settings/tokens"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your Hugging Face token (if needed)"
echo "2. Explore the examples in different directories:"
echo "   - Genel-1/ for basic transformer examples"
echo "   - Genel-2/ for vision transformers"  
echo "   - 'Multi Modal'/ for multimodal examples"
echo "   - llama/ for LLaMA implementation"
echo "3. Run: python test-time-scaling.py for a quick test"
echo ""
echo "📚 Check README.md for detailed usage instructions"