#!/bin/bash

# Start LiveKit Agent Service
# This script starts the agent service in development mode

echo "🚀 Starting LiveKit Agent Service..."
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo ""
    echo "Please create a .env file with the following variables:"
    echo "  LIVEKIT_URL=wss://your-project.livekit.cloud"
    echo "  LIVEKIT_API_KEY=your_api_key"
    echo "  LIVEKIT_API_SECRET=your_api_secret"
    echo "  OPENAI_API_KEY=your_openai_key"
    echo ""
    exit 1
fi

# Resolve Python (macOS often has python3 but not python)
PYTHON=""
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Error: Python not found!"
    echo "Install Python 3.10+ (e.g. brew install python@3.12) and ensure python3 is on your PATH."
    exit 1
fi

# Check if dependencies are installed
if ! "$PYTHON" -c "import livekit" &> /dev/null; then
    echo "⚠️  Dependencies not installed. Installing now..."
    echo ""
    
    if command -v uv &> /dev/null; then
        echo "Using uv to install dependencies..."
        uv pip install -r requirements.txt
    else
        echo "Using pip to install dependencies..."
        "$PYTHON" -m pip install -r requirements.txt
    fi
    
    echo ""
fi

echo "✅ Environment ready"
echo ""
echo "📡 Connecting to LiveKit..."
echo "   The agent will automatically join rooms when users start interviews"
echo ""
echo "Press Ctrl+C to stop the agent service"
echo ""
echo "─────────────────────────────────────────────────────────────"
echo ""

# Start the agent service
"$PYTHON" agent_service.py start


