# LiveKit Agent Service

Python service that creates LiveKit agents for conducting AI-powered interviews using **OpenAI's Realtime API**.

The agent automatically joins LiveKit rooms and conducts interviews with:

- **OpenAI Realtime API** for real-time voice conversation (handles STT, LLM, and TTS in one)
- **Dynamic prompt customization** from room metadata
- **Noise cancellation** for better audio quality

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended - faster)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with your credentials:

```bash
# LiveKit Configuration
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key_here
LIVEKIT_API_SECRET=your_livekit_api_secret_here

# OpenAI API Key (for Realtime API)
OPENAI_API_KEY=your_openai_api_key_here

# AWS S3 Configuration (for call recordings)
AWS_ACCESS_KEY_ID=your_aws_access_key_id_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key_here
AWS_REGION=us-east-1
AWS_S3_BUCKET_NAME=moktalk-call-recordings

# Optional: LiveKit Webhook URL for recording completion notifications
# LIVEKIT_WEBHOOK_URL=https://your-domain.com/api/livekit-webhook
```

### 3. Start the Service

```bash
# Easy way (using the start script)
./start-agent.sh

# Or manually
python agent_service.py dev
```

The agent will connect to LiveKit and automatically join interview rooms when users start interviews.

## How It Works

### Architecture

```
┌──────────────┐         ┌─────────────┐         ┌──────────────┐
│   Frontend   │────────▶│  LiveKit    │◀────────│    Agent     │
│   (User)     │         │   Cloud     │         │   Service    │
└──────────────┘         └─────────────┘         └──────────────┘
       │                        │                        │
       │ 1. Join room          │                        │
       │──────────────────────▶│                        │
       │                        │ 2. Notify agent       │
       │                        │───────────────────────▶│
       │                        │                        │
       │                        │ 3. Agent joins        │
       │                        │◀───────────────────────│
       │                        │                        │
       │ 4. Voice conversation  │                        │
       │◀──────────────────────▶│◀──────────────────────▶│
```

### Custom Prompt Flow

1. **Frontend creates room** with metadata containing custom prompt
2. **Agent detects new room** via LiveKit SDK
3. **Agent reads metadata** and extracts custom prompt
4. **Agent creates AI assistant** with custom instructions
5. **Agent joins room** and starts conversation using OpenAI Realtime API

### Real-time Conversation

The OpenAI Realtime API handles everything in one:

- **Speech-to-Text**: User speaks → automatically transcribed
- **LLM Processing**: Text → AI generates intelligent response
- **Text-to-Speech**: Response → natural voice output
- **All in real-time** with low latency

## Environment Variables

| Variable                | Description                                                    | Required                                          |
| ----------------------- | -------------------------------------------------------------- | ------------------------------------------------- |
| `LIVEKIT_URL`           | LiveKit WebSocket URL (e.g., wss://your-project.livekit.cloud) | ✅ Yes                                            |
| `LIVEKIT_API_KEY`       | LiveKit API Key                                                | ✅ Yes                                            |
| `LIVEKIT_API_SECRET`    | LiveKit API Secret                                             | ✅ Yes                                            |
| `OPENAI_API_KEY`        | OpenAI API Key for Realtime API                                | ✅ Yes                                            |
| `AWS_ACCESS_KEY_ID`     | AWS Access Key ID for S3 recordings                            | ✅ Yes                                            |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Access Key for S3 recordings                        | ✅ Yes                                            |
| `AWS_REGION`            | AWS Region (e.g., us-east-1)                                   | ⚠️ Optional (defaults to us-east-1)               |
| `AWS_S3_BUCKET_NAME`    | S3 Bucket name for storing recordings                          | ⚠️ Optional (defaults to moktalk-call-recordings) |
| `LIVEKIT_WEBHOOK_URL`   | Webhook URL for recording completion notifications             | ⚠️ Optional                                       |

## Customization

### Changing the Voice

Edit the `openai.realtime.RealtimeModel()` configuration in `agent_service.py`:

```python
session = AgentSession(
    llm=openai.realtime.RealtimeModel(
        voice="coral"  # Options: alloy, echo, fable, onyx, nova, shimmer, coral
    )
)
```

Available voices:

- **alloy**: Neutral and balanced
- **echo**: Warm and upbeat
- **fable**: British accent, expressive
- **onyx**: Deep and authoritative
- **nova**: Energetic and friendly
- **shimmer**: Soft and calm
- **coral**: Natural and conversational (default)

### Custom Instructions

The agent automatically reads custom instructions from room metadata. This is set by the frontend when creating the room:

```python
# Agent reads from room metadata
if ctx.room.metadata:
    metadata = json.loads(ctx.room.metadata)
    custom_instructions = metadata.get("customPrompt", "default...")

# Create assistant with custom instructions
agent = Assistant(instructions=custom_instructions)
```

### Noise Cancellation

The service includes noise cancellation by default. You can customize it:

```python
room_options=room_io.RoomOptions(
    audio_input=room_io.AudioInputOptions(
        noise_cancellation=lambda params: noise_cancellation.BVC(),
    ),
)
```

Options:

- `noise_cancellation.BVC()`: Basic Voice Cancellation (default)
- `noise_cancellation.BVCTelephony()`: Optimized for phone calls
- `None`: Disable noise cancellation

## Development

### Running in Development Mode

```bash
python agent_service.py dev
```

This will:

- Connect to LiveKit Cloud
- Wait for rooms to join
- Log all events to console
- Show when custom prompts are loaded

### Testing the Flow

1. **Start the agent service** (this terminal)

   ```bash
   ./start-agent.sh
   ```

2. **Start the frontend** (separate terminal)

   ```bash
   cd ../../moktalk
   npm run dev
   ```

3. **Open the app** in your browser

   - Navigate to http://localhost:3000/interview
   - Configure interview settings
   - Start interview

4. **Check agent logs** to see:
   ```
   ✅ Agent service ready, waiting for rooms...
   ✅ Room joined: interview-1234567890-abc123
   ✅ Using custom instructions from room metadata: You are conducting...
   ```

### Debugging

Enable verbose logging by checking the console output:

```
INFO:livekit.agents:Starting agent service...
INFO:livekit.agents:Connecting to LiveKit server at wss://...
INFO:livekit.agents:Agent service ready, waiting for rooms...
Using custom instructions from room metadata: You are conducting...
```

## Production Deployment

### Docker Deployment (Recommended)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run the agent
CMD ["python", "agent_service.py", "start"]
```

Build and run:

```bash
docker build -t moktalk-agent .
docker run -d --env-file .env moktalk-agent
```

### VPS Deployment

1. **Deploy to server**:

   ```bash
   scp -r . user@server:/opt/moktalk-agent
   ```

2. **Create systemd service** (`/etc/systemd/system/moktalk-agent.service`):

   ```ini
   [Unit]
   Description=Moktalk LiveKit Agent Service
   After=network.target

   [Service]
   Type=simple
   User=moktalk
   WorkingDirectory=/opt/moktalk-agent
   Environment="PATH=/opt/moktalk-agent/venv/bin"
   ExecStart=/opt/moktalk-agent/venv/bin/python agent_service.py start
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

3. **Start service**:
   ```bash
   sudo systemctl enable moktalk-agent
   sudo systemctl start moktalk-agent
   ```

### Platform Deployment

Deploy to platforms like:

- **Railway**: Connect GitHub repo, set env vars, deploy
- **Render**: Create web service, set env vars, deploy
- **DigitalOcean App Platform**: Similar process

**Important**: Ensure the platform supports:

- WebSocket connections
- Long-running processes
- Python 3.10+

## Troubleshooting

### Agent Not Starting

**Problem**: Service crashes on startup

**Solutions**:

- Check environment variables are set correctly in `.env`
- Verify LiveKit credentials (API key, secret, URL)
- Ensure OpenAI API key is valid
- Check Python version: `python --version` (need 3.10+)

### Agent Not Joining Rooms

**Problem**: Rooms created but agent doesn't join

**Solutions**:

- Verify agent service is running: `ps aux | grep agent_service`
- Check LiveKit URL matches between frontend and agent
- Ensure API keys match
- Check firewall/network settings
- Look for errors in agent logs

### Custom Prompt Not Working

**Problem**: Agent uses default prompt instead of custom one

**Solutions**:

- Check room metadata is being set (check frontend logs)
- Verify agent is reading metadata (check agent logs for "Using custom instructions...")
- Ensure JSON parsing is working (check for JSON errors)
- Verify metadata keys match: `customPrompt`, `greetingInstructions`

### Audio Issues

**Problem**: No audio or poor quality

**Solutions**:

- Verify OpenAI API key is valid and has Realtime API access
- Check OpenAI Realtime API quota and billing
- Ensure noise cancellation is configured correctly
- Test with different voice options
- Check browser microphone permissions
- Verify audio tracks are being published

### Connection Timeout

**Problem**: Agent connects but conversation doesn't start

**Solutions**:

- Check OpenAI API status
- Verify network connectivity
- Look for errors in agent logs
- Try restarting the agent service

## Advanced Features

### Custom Greeting

The greeting is controlled by `greetingInstructions` in room metadata. Modify in the frontend API:

```typescript
const roomMetadata = {
  customPrompt: "...",
  greetingInstructions:
    "Start by asking about their background in a friendly tone.",
  // ...
};
```

### Context Awareness

Use document context from metadata to make the agent aware of candidate information:

```python
if "documentContext" in metadata:
    context = metadata["documentContext"]
    instructions = f"{custom_instructions}\n\nCandidate context: {context}"
```

### Multiple Agent Types

Create different agent classes for different interview types:

```python
class TechnicalInterviewer(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a technical interviewer...")

class BehavioralInterviewer(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a behavioral interviewer...")

# Use based on metadata
agent_type = metadata.get("interviewType", "technical")
if agent_type == "technical":
    agent = TechnicalInterviewer()
else:
    agent = BehavioralInterviewer()
```

## Monitoring

### Health Checks

Monitor the agent service by:

- Checking process status: `ps aux | grep agent_service`
- Monitoring logs for errors
- Using LiveKit dashboard to see active agents
- Setting up alerts for crashes

### Important Log Messages

```
✅ Agent service ready, waiting for rooms...
✅ Room joined: interview-1234567890-abc123
✅ Using custom instructions from room metadata
❌ Error parsing room metadata
❌ Error connecting to LiveKit
```

### LiveKit Dashboard

Use the LiveKit Cloud dashboard to:

- View active rooms and participants
- See connected agents
- Monitor connection quality and bandwidth
- Debug connection issues
- View room metadata

## Code Reference

### Main Agent Handler

```python
@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    # Read custom prompt from room metadata
    custom_instructions = "You are a helpful voice AI assistant."
    greeting_instructions = "Greet the user..."

    if ctx.room.metadata:
        metadata = json.loads(ctx.room.metadata)
        custom_instructions = metadata.get("customPrompt", custom_instructions)
        greeting_instructions = metadata.get("greetingInstructions", greeting_instructions)

    # Create session with OpenAI Realtime API
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="coral")
    )

    # Start session with custom agent
    await session.start(
        room=ctx.room,
        agent=Assistant(instructions=custom_instructions),
        room_options=room_io.RoomOptions(...)
    )

    # Generate initial greeting
    await session.generate_reply(instructions=greeting_instructions)
```

## Resources

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- [LiveKit Python SDK](https://github.com/livekit/python-sdks)
- [Main Project Setup Guide](../../LIVEKIT_SETUP_GUIDE.md)
- [Call Flow Documentation](../../CALL_FLOW.md)

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the [main setup guide](../../LIVEKIT_SETUP_GUIDE.md)
3. Check [call flow documentation](../../CALL_FLOW.md)
4. Review LiveKit and OpenAI documentation
5. Open an issue in the project repository
