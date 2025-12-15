import json
import os
import uuid
import asyncio
import threading
from datetime import datetime
from dotenv import load_dotenv

from livekit import agents, rtc, api
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.plugins import (
    openai,
    noise_cancellation,
)
from fastapi import FastAPI
import uvicorn

load_dotenv(".env")

# Store room reference for data channel communication
current_room = None
# Store transcriptions as they come in
transcription_history = []
# Store egress info for recording tracking
egress_info_storage = {}


async def send_transcription(
    ctx: agents.JobContext,
    participant: rtc.Participant,
    track_sid: str,
    segment_id: str,
    text: str,
    is_final: bool = True,
    start_time: float = 0,
    end_time: float = 0,
):
    """Publish transcription using LiveKit's transcription API"""
    # Convert seconds to nanoseconds (integers) for LiveKit API
    start_time_ns = int(start_time * 1_000_000_000) if start_time > 0 else 0
    end_time_ns = int(end_time * 1_000_000_000) if end_time > 0 else int((start_time + 1.0) * 1_000_000_000)
    
    transcription = rtc.Transcription(
        participant_identity=participant.identity,
        track_sid=track_sid,
        segments=[
            rtc.TranscriptionSegment(
                id=segment_id,
                text=text,
                start_time=start_time_ns,
                end_time=end_time_ns,
                language="en",
                final=is_final,
            )
        ],
    )
    await ctx.room.local_participant.publish_transcription(transcription)
    
    # Store in history for final transcript
    transcription_history.append({
        "role": "user" if participant.identity != ctx.room.local_participant.identity else "agent",
        "message": text,
        "timestamp": datetime.now().isoformat(),
        "is_final": is_final,
    })


class Assistant(Agent):
    def __init__(self, instructions: str = "You are a helpful voice AI assistant.") -> None:
        super().__init__(instructions=instructions)

server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    global current_room, transcription_history
    
    print(f"\n{'='*60}")
    print(f"🤖 Agent detected room: {ctx.room.name}")
    print(f"{'='*60}")
    
    # Clear transcription history for new session
    transcription_history = []
    
    # Store room reference
    current_room = ctx.room
    
    # Connect to the room first
    await ctx.connect()
    print("✓ Agent connected to room")
    
    # Start recording to S3
    try:
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-2")
        s3_bucket_name = os.getenv("AWS_S3_BUCKET_NAME", "moktalk-call-recordings")
        livekit_url = os.getenv("LIVEKIT_URL")
        livekit_api_key = os.getenv("LIVEKIT_API_KEY")
        livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
        webhook_url = os.getenv("LIVEKIT_WEBHOOK_URL")
        
        if aws_access_key_id and aws_secret_access_key and livekit_url and livekit_api_key and livekit_api_secret:
            print(f"\n{'='*60}")
            print("🎥 Starting recording to S3...")
            print(f"{'='*60}")
            
            # Initialize LiveKit API client
            lkapi = api.LiveKitAPI(
                url=livekit_url.replace("wss://", "https://").replace("ws://", "http://"),
                api_key=livekit_api_key,
                api_secret=livekit_api_secret,
            )
            
            # Create file output path in organized folder structure
            # Format: moktalk/recordings/{conversationId}/recording.mp4
            conversation_id = ctx.room.name
            filepath = f"moktalk/recordings/{conversation_id}/recording.mp4"
            
            # Create S3 upload configuration
            s3_upload = api.S3Upload(
                access_key=aws_access_key_id,
                secret=aws_secret_access_key,
                region=aws_region,
                bucket=s3_bucket_name,
            )
            
            # Create encoded file output
            file_output = api.EncodedFileOutput(
                filepath=filepath,
                s3=s3_upload,
            )
            
            # Start room composite egress
            egress_request = api.RoomCompositeEgressRequest(
                room_name=ctx.room.name,
                file=file_output,
            )
            
            # Note: Webhook configuration would be set via webhooks parameter if needed
            # For now, we'll rely on LiveKit Cloud webhook configuration
            
            egress_info = await lkapi.egress.start_room_composite_egress(egress_request)
            
            # Store egress info for later status checking
            egress_info_storage[ctx.room.name] = {
                "egress_id": egress_info.egress_id,
                "s3_path": filepath,
                "lkapi": lkapi,  # Store API client for status checking
            }
            
            print(f"✓ Recording started successfully")
            print(f"  Egress ID: {egress_info.egress_id}")
            print(f"  S3 Path: {filepath}")
            print(f"  Status: {egress_info.status}")
            print(f"{'='*60}\n")
        else:
            print("⚠️  AWS or LiveKit credentials not configured, skipping recording setup")
            missing = []
            if not aws_access_key_id: missing.append("AWS_ACCESS_KEY_ID")
            if not aws_secret_access_key: missing.append("AWS_SECRET_ACCESS_KEY")
            if not livekit_url: missing.append("LIVEKIT_URL")
            if not livekit_api_key: missing.append("LIVEKIT_API_KEY")
            if not livekit_api_secret: missing.append("LIVEKIT_API_SECRET")
            if missing:
                print(f"  Missing: {', '.join(missing)}")
    except Exception as recording_error:
        print(f"⚠️  Failed to start recording: {recording_error}")
        import traceback
        traceback.print_exc()
        print("  Continuing without recording...\n")
    
    # Default instructions - will be updated from AccessToken metadata
    custom_instructions = "You are a helpful voice AI assistant."
    greeting_instructions = "Greet the user and offer your assistance. You should start by speaking in English."
    
    # Wait for participant to join and read their metadata
    try:
        print("⏳ Waiting for participant to join...")
        participant = await ctx.wait_for_participant()
        
        if participant.metadata:
            metadata = json.loads(participant.metadata)
            prompt = metadata.get("prompt", "")
            
            if prompt:
                custom_instructions = prompt
                print(f"\n{'='*60}")
                print(f"✅ Prompt read from AccessToken metadata")
                print(f"{'='*60}")
                print(f"Prompt: {custom_instructions[:200]}...")
                print(f"{'='*60}\n")
            else:
                print("⚠️  No prompt found in participant metadata, using default instructions")
        else:
            print("⚠️  No metadata found on participant, using default instructions")
    except Exception as e:
        print(f"⚠️  Error waiting for participant or reading metadata: {e}")
        print("Using default instructions")
    
    # Create session with the instructions from AccessToken metadata or default
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="coral"
        )
    )

    # Store session start time for calculating duration
    session_start_time = datetime.now()
    session_start_timestamp = datetime.now().timestamp()
    
    # Helper function to handle user transcription asynchronously
    async def handle_user_transcription_async(ev, ctx_ref, start_ts):
        """Handle user transcription events asynchronously"""
        try:
            text = getattr(ev, 'text', '') or getattr(ev, 'transcript', '')
            if text and text.strip():
                print(f"👤 User transcription: {text[:100]}...")
                
                # Get user participant
                user_participant = None
                for participant in ctx_ref.room.remote_participants.values():
                    if participant.identity != ctx_ref.room.local_participant.identity:
                        user_participant = participant
                        break
                
                if user_participant:
                    # Get track SID (use a default or get from event if available)
                    track_sid = getattr(ev, 'track_sid', 'user-audio-track')
                    segment_id = str(uuid.uuid4())
                    
                    # Calculate relative time
                    current_time = datetime.now().timestamp()
                    relative_start = current_time - start_ts
                    relative_end = relative_start + 1.0  # Approximate 1 second duration
                    
                    # Publish transcription
                    await send_transcription(
                        ctx=ctx_ref,
                        participant=user_participant,
                        track_sid=track_sid,
                        segment_id=segment_id,
                        text=text.strip(),
                        is_final=True,
                        start_time=relative_start,
                        end_time=relative_end,
                    )
                    
                    # Trigger sending transcript to frontend (debounced)
                    if send_transcript_ref.get("func"):
                        asyncio.create_task(send_transcript_ref["func"]())
        except Exception as e:
            print(f"⚠️  Error handling user transcription: {e}")
            import traceback
            traceback.print_exc()
    
    # Helper function to handle conversation items asynchronously
    async def handle_conversation_item_async(ev, ctx_ref, start_ts):
        """Handle conversation items being added asynchronously"""
        try:
            item = getattr(ev, 'item', None)
            if not item:
                return
            
            # Extract text and role from conversation item
            text = ""
            role = "agent"
            
            if hasattr(item, 'content'):
                if isinstance(item.content, str):
                    text = item.content
                elif isinstance(item.content, list):
                    text = " ".join(
                        part.get('text', '') if isinstance(part, dict) else str(part)
                        for part in item.content
                    )
            
            if hasattr(item, 'role'):
                role = item.role
            
            if text and text.strip():
                # Determine if this is user or agent
                is_user = role == "user" or (hasattr(item, 'role') and item.role == "user")
                
                print(f"{'👤' if is_user else '🤖'} Conversation item ({'user' if is_user else 'agent'}): {text[:100]}...")
                
                # Get the appropriate participant
                if is_user:
                    participant = None
                    for p in ctx_ref.room.remote_participants.values():
                        if p.identity != ctx_ref.room.local_participant.identity:
                            participant = p
                            break
                    if not participant:
                        return
                else:
                    participant = ctx_ref.room.local_participant
                
                # Calculate relative time
                current_time = datetime.now().timestamp()
                relative_start = current_time - start_ts
                relative_end = relative_start + 1.0
                
                # Publish transcription
                track_sid = 'user-audio-track' if is_user else 'agent-audio-track'
                segment_id = str(uuid.uuid4())
                
                await send_transcription(
                    ctx=ctx_ref,
                    participant=participant,
                    track_sid=track_sid,
                    segment_id=segment_id,
                    text=text.strip(),
                    is_final=True,
                    start_time=relative_start,
                    end_time=relative_end,
                )
                
                # Trigger sending transcript to frontend (debounced)
                if send_transcript_ref.get("func"):
                    asyncio.create_task(send_transcript_ref["func"]())
        except Exception as e:
            print(f"⚠️  Error handling conversation item: {e}")
            import traceback
            traceback.print_exc()
    
    # Add shutdown callback to check recording status and save transcript
    async def check_recording_status():
        """Check if recording completed successfully"""
        # Wait a moment for egress to finalize and upload
        await asyncio.sleep(2)
        
        room_name = ctx.room.name
        if room_name in egress_info_storage:
            egress_data = egress_info_storage[room_name]
            egress_id = egress_data["egress_id"]
            lkapi = egress_data["lkapi"]
            s3_path = egress_data["s3_path"]
            s3_bucket_name = os.getenv("AWS_S3_BUCKET_NAME", "moktalk-call-recordings")
            
            try:
                print(f"\n{'='*60}")
                print("🔍 Checking recording status...")
                print(f"{'='*60}")
                
                # Get egress status using list_egress
                from livekit.api import ListEgressRequest
                list_request = ListEgressRequest()
                list_request.room_name = room_name
                egress_list_response = await lkapi.egress.list_egress(list_request)
                
                egress_status = None
                if hasattr(egress_list_response, 'items') and egress_list_response.items:
                    for egress in egress_list_response.items:
                        if egress.egress_id == egress_id:
                            egress_status = egress
                            break
                
                print(f"  Egress ID: {egress_id}")
                print(f"  S3 Path: {s3_path}")
                
                if not egress_status:
                    print(f"⚠️  Could not find egress with ID: {egress_id}")
                    print(f"   This might mean:")
                    print(f"   - Recording is still processing/uploading")
                    print(f"   - Recording failed silently")
                    print(f"   - Check LiveKit Cloud dashboard for egress status")
                    print(f"   - Check S3 bucket: {s3_bucket_name}")
                    print(f"{'='*60}\n")
                    return
                
                print(f"  Status: {egress_status.status}")
                
                if egress_status.file_results and len(egress_status.file_results) > 0:
                    file_result = egress_status.file_results[0]
                    print(f"  File Location: {file_result.location}")
                    print(f"  File Size: {file_result.size_bytes} bytes")
                    print(f"✓ Recording completed and uploaded to S3")
                elif egress_status.status == "EGRESS_COMPLETE":
                    print(f"✓ Recording completed (checking file results...)")
                elif egress_status.status == "EGRESS_ACTIVE":
                    print(f"⚠️  Recording still in progress")
                elif egress_status.status == "EGRESS_FAILED":
                    print(f"❌ Recording failed: {egress_status.error}")
                else:
                    print(f"ℹ️  Recording status: {egress_status.status}")
                
                print(f"{'='*60}\n")
                
                # Clean up - properly close LiveKit API client
                try:
                    await lkapi.aclose()
                    print("✓ LiveKit API client closed successfully")
                except Exception as close_error:
                    print(f"⚠️  Error closing LiveKit API client: {close_error}")
                
                # Remove from storage
                if room_name in egress_info_storage:
                    del egress_info_storage[room_name]
                
            except Exception as status_error:
                print(f"⚠️  Error checking recording status: {status_error}")
                import traceback
                traceback.print_exc()
                
                # Ensure we close the API client even on error
                try:
                    await lkapi.aclose()
                except Exception:
                    pass
                
                # Clean up storage
                if room_name in egress_info_storage:
                    del egress_info_storage[room_name]
    
    # Add shutdown callback to save transcript
    async def save_transcript():
        print(f"\n{'='*60}")
        print("💾 Collecting conversation transcript...")
        print(f"{'='*60}")
        
        try:
            # Collect transcript from multiple sources
            transcript_items = []
            
            # First, use transcription_history (from published transcriptions)
            if transcription_history:
                print(f"✓ Found {len(transcription_history)} items in transcription history")
                for item in transcription_history:
                    if item.get("is_final", True):  # Only include final transcriptions
                        transcript_items.append({
                            "role": item["role"],
                            "message": item["message"]
                        })
            
            # Also collect from session history as fallback
            # session.history is a ChatContext object, try to convert to dict/list
            session_items_count = 0
            try:
                # Try to convert history to dict or list
                history_data = None
                if hasattr(session.history, 'to_dict'):
                    history_data = session.history.to_dict()
                elif hasattr(session.history, 'items'):
                    history_data = session.history.items
                elif hasattr(session.history, 'messages'):
                    history_data = session.history.messages
                
                if history_data:
                    # Handle dict format
                    if isinstance(history_data, dict):
                        items = history_data.get('items', history_data.get('messages', []))
                    elif isinstance(history_data, list):
                        items = history_data
                    else:
                        items = []
                    
                    for item in items:
                        # Handle both dict and object formats
                        if isinstance(item, dict):
                            role = item.get('role', 'agent')
                            content = item.get('content', item.get('text', ''))
                        else:
                            # Object format
                            role = getattr(item, 'role', 'agent')
                            content = getattr(item, 'content', getattr(item, 'text', ''))
                        
                        # Extract message text
                        message = ""
                        if isinstance(content, str):
                            message = content
                        elif isinstance(content, list):
                            message = " ".join(
                                part.get('text', '') if isinstance(part, dict) else str(part)
                                for part in content
                            )
                        else:
                            message = str(content)
                        
                        if message.strip():
                            # Normalize role
                            role_normalized = "user" if role == "user" else "agent"
                            
                            # Check if we already have this in transcription_history
                            is_duplicate = any(
                                t["message"] == message.strip() and t["role"] == role_normalized
                                for t in transcript_items
                            )
                            
                            if not is_duplicate:
                                transcript_items.append({
                                    "role": role_normalized,
                                    "message": message.strip()
                                })
                                session_items_count += 1
            except Exception as history_error:
                print(f"⚠️  Could not access session history: {history_error}")
                print("    Using transcription_history only")
                import traceback
                traceback.print_exc()
            
            if session_items_count > 0:
                print(f"✓ Added {session_items_count} additional items from session history")
            
            if transcript_items:
                print(f"✓ Collected {len(transcript_items)} transcript items")
                print("\nTranscript preview:")
                for i, item in enumerate(transcript_items[:3]):  # Show first 3 items
                    print(f"  {i+1}. [{item['role']}]: {item['message'][:100]}...")
                
                # Calculate session duration
                session_end_time = datetime.now()
                duration_seconds = int((session_end_time - session_start_time).total_seconds())
                
                print(f"\n✓ Session duration: {duration_seconds} seconds")
                print(f"✓ Room: {ctx.room.name}")
                
                # Save to file for debugging
                transcript_filename = f"/tmp/transcript_{ctx.room.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                transcript_data = {
                    "room": ctx.room.name,
                    "duration": duration_seconds,
                    "timestamp": datetime.now().isoformat(),
                    "transcript": transcript_items
                }
                
                try:
                    with open(transcript_filename, 'w') as f:
                        json.dump(transcript_data, f, indent=2)
                    print(f"✓ Transcript saved locally to: {transcript_filename}")
                except Exception as file_error:
                    print(f"⚠️  Could not save transcript to file: {file_error}")
                
                # Note: Transcript is sent via data channel when participant disconnects
                # (handled by on_participant_disconnected callback)
                
                print(f"{'='*60}")
                print("✓ Transcript collection complete")
                print(f"{'='*60}\n")
            else:
                print("⚠️  No transcript items found")
                
        except Exception as e:
            print(f"❌ Error saving transcript: {e}")
            import traceback
            traceback.print_exc()
    
    # Flag to track last transcript send time (for debouncing)
    last_transcript_send = {"time": 0}
    
    # Function to send transcript via data channel
    async def send_transcript_to_frontend(force: bool = False):
        """Send collected transcript to frontend via data channel"""
        try:
            # Collect transcript items from transcription_history
            transcript_items = []
            for item in transcription_history:
                if item.get("is_final", True):
                    transcript_items.append({
                        "role": item["role"],
                        "message": item["message"]
                    })
            
            if not transcript_items:
                return  # No items to send
            
            # Debounce: don't send more than once every 5 seconds unless forced
            current_time = datetime.now().timestamp()
            if not force and (current_time - last_transcript_send["time"]) < 5:
                return
            
            # Calculate duration
            duration_seconds = int((datetime.now() - session_start_time).total_seconds())
            
            transcript_data = {
                "room": ctx.room.name,
                "duration": duration_seconds,
                "timestamp": datetime.now().isoformat(),
                "transcript": transcript_items
            }
            
            # Try to send via data channel
            if current_room:
                participants = list(current_room.remote_participants.values())
                
                if participants:
                    print(f"📤 Sending transcript to {len(participants)} participant(s)... ({len(transcript_items)} items)")
                    data_message = json.dumps({
                        "type": "transcript",
                        "data": transcript_data
                    })
                    
                    await current_room.local_participant.publish_data(
                        data_message.encode('utf-8'),
                        reliable=True
                    )
                    last_transcript_send["time"] = current_time
                    print(f"✓ Transcript sent via data channel")
                else:
                    print("⚠️  No participants available to send transcript to")
        except Exception as e:
            print(f"⚠️  Error sending transcript: {e}")
            import traceback
            traceback.print_exc()
    
    # Store reference to send function so handlers can call it
    send_transcript_ref = {"func": None}
    
    # Register the shutdown callbacks
    ctx.add_shutdown_callback(check_recording_status)
    ctx.add_shutdown_callback(save_transcript)
    
    # Also try to send transcript on shutdown (force send)
    async def send_on_shutdown():
        await send_transcript_to_frontend(force=True)
    ctx.add_shutdown_callback(send_on_shutdown)
    
    # Set the reference so handlers can access it
    send_transcript_ref["func"] = send_transcript_to_frontend

    await session.start(
        room=ctx.room,
        agent=Assistant(instructions=custom_instructions),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )
    
    # Set up transcription event listeners after session starts
    # Use synchronous callbacks that create async tasks
    @session.on("user_input_transcribed")
    def on_user_transcription(ev):
        """Handle user transcription events - synchronous wrapper"""
        asyncio.create_task(handle_user_transcription_async(ev, ctx, session_start_timestamp))
    
    @session.on("conversation_item_added")
    def on_conversation_item(ev):
        """Handle conversation items - synchronous wrapper"""
        asyncio.create_task(handle_conversation_item_async(ev, ctx, session_start_timestamp))

    await session.generate_reply(
        instructions=greeting_instructions
    )


# Health check server for Railway deployment
health_app = FastAPI()

@health_app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {"status": "healthy", "service": "moktalk-agent"}

@health_app.get("/")
async def root():
    """Root endpoint"""
    return {"status": "ok", "service": "moktalk-agent"}

def run_health_server():
    """Run health check server in background thread"""
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(health_app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    # Start health check server in background thread for Railway
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    print(f"✅ Health check server started on port {os.getenv('PORT', 8080)}")
    
    # Start the agent service
    agents.cli.run_app(server)