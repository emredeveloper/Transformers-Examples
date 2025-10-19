#!/bin/bash
set -euo pipefail

trap 'echo "❌ An unexpected error occurred. Exiting setup." >&2' ERR
trap 'echo "⚠️ Setup interrupted." >&2' INT TERM

# Transformers Examples Setup Script
echo "🚀 Setting up Transformers Examples repository..."

# Suggest Windows alternative if running in Windows-like environment
if [[ "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* || "${OSTYPE:-}" == win32* ]]; then
    echo "ℹ️  Windows users can run requirements.bat or adapt the steps in a PowerShell script."
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

python_version="$(python3 --version)"
echo "✅ Python 3 found: ${python_version}"

# Create virtual environment (optional but recommended)
if [[ "${1:-}" == "--venv" ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    # shellcheck source=/dev/null
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Install requirements
echo "📥 Installing dependencies..."
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "⚠️  Warning: No active virtual environment detected. Consider running with --venv or activating one manually."
fi
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Check if .env file exists
if [[ ! -f ".env" ]]; then
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
echo "3. Run: python3 test-time-scaling.py for a quick test"
echo ""
echo "📚 Check README.md for detailed usage instructions"

