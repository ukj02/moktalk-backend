# Quick Start Guide

## Prerequisites

You mentioned you already have:
- ✅ Deepgram API key
- ✅ Cartesia API key

You also need:
- LiveKit credentials (URL, API Key, API Secret)
- OpenAI API key

## Step 1: Install Dependencies

```bash
cd services/agent

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirements.txt
```

## Step 2: Configure Environment Variables

Create a `.env` file in `services/agent/`:

```bash
# LiveKit Configuration (from your LiveKit dashboard)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# OpenAI (for conversational AI)
OPENAI_API_KEY=your_openai_api_key

# Deepgram (for speech-to-text) - YOU HAVE THIS
DEEPGRAM_API_KEY=your_deepgram_api_key

# Cartesia (for text-to-speech) - YOU HAVE THIS
CARTESIA_API_KEY=your_cartesia_api_key
```

## Step 3: Start the Agent

```bash
# From services/agent directory
python agent_service.py dev
```

You should see:
```
============================================================
🎙️  LiveKit Interview Agent Service
============================================================
LiveKit URL: wss://your-project.livekit.cloud
OpenAI Model: gpt-4o-mini
STT Provider: Deepgram (nova-2-general)
TTS Provider: Cartesia
============================================================
✓ All required environment variables are set
🔥 Prewarming agent plugins...
✓ Prewarm complete
```

## Step 4: Test from Frontend

1. Start your Next.js frontend (in another terminal):
   ```bash
   cd moktalk
   npm run dev
   ```

2. Go to the interview page and click "Start Interview"

3. The agent should automatically join the room and greet you!

## Expected Flow

1. **Frontend**: User clicks "Start Interview"
2. **Frontend**: Calls `/api/create-web-call` to get access token
3. **Frontend**: Connects to LiveKit room
4. **Agent**: Detects new room and automatically joins
5. **Agent**: Says "Hello! Welcome to your interview session..."
6. **User**: Speaks (microphone → Deepgram → text)
7. **Agent**: Thinks (text → OpenAI → response)
8. **Agent**: Speaks (response → Cartesia → audio)
9. **Conversation continues...**

## Troubleshooting

### "Missing required environment variables"
- Check your `.env` file exists in `services/agent/`
- Verify all variables are set (no empty values)

### Agent doesn't join the room
- Ensure agent service is running (`python agent_service.py dev`)
- Check that LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET match your LiveKit dashboard
- Look for errors in the agent terminal

### Can't hear the agent
- Check browser console for audio errors
- Verify CARTESIA_API_KEY is correct
- Ensure browser has permission to play audio

### Agent can't hear you
- Check microphone permissions in browser
- Verify DEEPGRAM_API_KEY is correct
- Look at the agent logs to see if it's receiving audio

## Next Steps

Once it's working:
- Customize the interviewer prompt in `agent_service.py`
- Change the voice by modifying the Cartesia voice ID
- Adjust the LLM model (gpt-4o-mini, gpt-4o, etc.)
- Add custom logic for different interview types




