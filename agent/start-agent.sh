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

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found!"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

# Check if dependencies are installed
if ! python -c "import livekit" &> /dev/null; then
    echo "⚠️  Dependencies not installed. Installing now..."
    echo ""
    
    if command -v uv &> /dev/null; then
        echo "Using uv to install dependencies..."
        uv pip install -r requirements.txt
    else
        echo "Using pip to install dependencies..."
        pip install -r requirements.txt
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
python agent_service.py start


