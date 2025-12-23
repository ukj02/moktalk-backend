import json
import os
import uuid
import asyncio
import threading
import time
import base64
from datetime import datetime
from typing import AsyncIterable, Optional, List
from dotenv import load_dotenv

from livekit import agents, rtc, api
from livekit.agents import (
    AgentServer, 
    AgentSession, 
    Agent, 
    room_io,
    llm,
    ChatContext,
    ChatMessage,
    FunctionTool,
    ModelSettings,
)
from livekit.agents.llm import ImageContent
from livekit.agents.utils import images
from livekit.plugins import (
    openai,
    noise_cancellation,
)
from fastapi import FastAPI
import uvicorn
import posthog

load_dotenv(".env")

# Initialize PostHog for LLM tracking

from posthog import Posthog

posthog_api_key = os.getenv("POSTHOG_API_KEY")
posthog_host = os.getenv("POSTHOG_HOST", "https://app.posthog.com")
posthog_client = None
if posthog_api_key:
    posthog_client = posthog.Posthog(
        project_api_key=posthog_api_key,
        host=posthog_host
    )
    print("✓ PostHog initialized for LLM tracking")
else:
    print("⚠️  POSTHOG_API_KEY not configured, LLM tracking disabled")
    
# Store room reference for data channel communication
current_room = None
# Store transcriptions as they come in
transcription_history = []
# Store egress info for recording tracking
egress_info_storage = {}
# Store latest video frame for multimodal context
latest_screen_frame = {"frame": None, "timestamp": 0}


def track_llm_generation(
    model: str,
    input_text: str,
    output_text: str,
    user_id: Optional[str] = None,
    room_name: Optional[str] = None,
    latency_ms: Optional[float] = None,
    token_usage: Optional[dict] = None,
    error: Optional[str] = None,
):
    """Track LLM generation event in PostHog"""
    if not posthog_client:
        return

    try:
        distinct_id = user_id or room_name or "anonymous"

        # PostHog LLM analytics expects $ai_generation events
        properties = {
            "$ai_model": model,
            "$ai_model_version": model,
            "$ai_provider": "openai",
            "$ai_input": input_text[:1000] if input_text else None,  # Limit length
            "$ai_output": output_text[:1000] if output_text else None,
            "$ai_input_tokens": token_usage.get("input_tokens") if token_usage else None,
            "$ai_output_tokens": token_usage.get("output_tokens") if token_usage else None,
            "$ai_total_tokens": token_usage.get("total_tokens") if token_usage else None,
            "$ai_latency_ms": latency_ms,
            "room_name": room_name,
            "session_type": "livekit_realtime",
        }

        if error:
            properties["error"] = error

        # Remove None values
        properties = {k: v for k, v in properties.items() if v is not None}

        posthog_client.capture(
            distinct_id=distinct_id,
            event="$ai_generation",
            properties=properties,
        )

        # Flush to ensure event is sent
        posthog_client.flush()
    except Exception as e:
        print(f"⚠️  Error tracking LLM generation: {e}")


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
    def __init__(self, instructions: str = "You are a helpful voice AI assistant.", room: rtc.Room = None) -> None:
        self._video_stream = None
        self._tasks = []
        self._room = room
        super().__init__(instructions=instructions)
    
    async def on_enter(self):
        """Called when agent joins the room - set up video tracking"""
        if not self._room:
            print("⚠️  Room not available in Assistant, skipping video setup")
            return
        
        # Watch for screen share tracks from remote participants
        @self._room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            # Check if this is a screen share track - use SOURCE_SCREEN_SHARE_VIDEO for video tracks
            if publication.source == rtc.TrackSource.SOURCE_SCREEN_SHARE_VIDEO and track.kind == rtc.TrackKind.KIND_VIDEO:
                print(f"🖥️  Screen share track detected, starting video stream capture")
                self._create_video_stream(track)
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Called when user finishes speaking - add latest screen frame to context"""
        global latest_screen_frame
        
        # If we have a recent screen frame, add it to the conversation context
        if latest_screen_frame["frame"] is not None:
            # Check if frame is recent (within last 5 seconds)
            current_time = time.time()
            if current_time - latest_screen_frame["timestamp"] < 5:
                print(f"📸 Adding screen frame to conversation context (age: {current_time - latest_screen_frame['timestamp']:.1f}s)")
                
                # Add the frame as visual context
                new_message.content.append(
                    ImageContent(image=latest_screen_frame["frame"])
                )
                
                # Also add a text prompt to analyze the screen
                if isinstance(new_message.content[0], str):
                    # If first item is text, enhance it
                    original_text = new_message.content[0]
                    new_message.content[0] = f"{original_text}\n\n[Note: You can see the candidate's screen share in the attached image. Consider what they're working on when responding.]"
    
    def _create_video_stream(self, track: rtc.Track):
        """Create video stream to capture frames from screen share"""
        global latest_screen_frame
        
        # Close any existing stream
        if self._video_stream is not None:
            self._video_stream.close()
        
        # Create new video stream
        self._video_stream = rtc.VideoStream(track)
        
        async def read_stream():
            """Continuously read and buffer the latest frame"""
            frame_count = 0
            async for event in self._video_stream:
                frame_count += 1
                
                # Sample every 30 frames (roughly once per second at 30fps)
                if frame_count % 30 == 0:
                    try:
                        # Encode frame to JPEG with reasonable size
                        image_bytes = images.encode(
                            event.frame,
                            images.EncodeOptions(
                                format="JPEG",
                                resize_options=images.ResizeOptions(
                                    width=1024,
                                    height=1024,
                                    strategy="scale_aspect_fit"
                                ),
                                quality=85
                            )
                        )
                        
                        # Store as base64 data URL
                        latest_screen_frame["frame"] = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                        latest_screen_frame["timestamp"] = time.time()
                        
                        if frame_count == 30:  # Log only on first successful capture
                            print(f"✓ Screen frame captured and encoded successfully")
                    except Exception as e:
                        print(f"⚠️  Error encoding video frame: {e}")
        
        # Start the stream reading task
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)

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
    
    # Log all existing participants and their tracks
    print(f"\n📊 Room state after connection:")
    print(f"  Remote participants: {len(ctx.room.remote_participants)}")
    for remote_participant in ctx.room.remote_participants.values():
        print(f"  - Participant: {remote_participant.identity}")
        print(f"    Track publications: {len(remote_participant.track_publications)}")
        for pub in remote_participant.track_publications.values():
            print(f"      {pub.kind} (source: {pub.source}, subscribed: {pub.subscribed}, track: {pub.track is not None})")
    print()
    
    # Debug: Track audio frames received
    received_audio_frames = {"count": 0, "logged": False}
    
    # Helper function to set up audio frame listener for a track
    def setup_audio_listener(track: rtc.Track, participant: rtc.RemoteParticipant):
        """Set up audio frame listener for debugging"""
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"🎤 Setting up audio frame listener for {participant.identity}")
            audio_stream = rtc.AudioStream(track)
            
            async def count_audio_frames():
                try:
                    async for frame_event in audio_stream:
                        received_audio_frames["count"] += 1
                        if not received_audio_frames["logged"] and received_audio_frames["count"] >= 10:
                            print(f"✓ ✓ ✓ RECEIVING AUDIO FRAMES from user! (count: {received_audio_frames['count']})")
                            received_audio_frames["logged"] = True
                except Exception as e:
                    print(f"⚠️  Error in audio frame listener: {e}")
            
            asyncio.create_task(count_audio_frames())
    
    # Add track subscription handler (consolidated - handles both debugging and audio frame setup)
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant
    ):
        print(f"📡 Track subscribed:")
        print(f"  - Participant: {participant.identity}")
        print(f"  - Track Kind: {track.kind}")
        print(f"  - Track Source: {publication.source}")
        print(f"  - Track SID: {track.sid}")
        print(f"  - Muted: {publication.muted}")
        
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            if publication.source == rtc.TrackSource.SOURCE_MICROPHONE:
                print(f"🎤 ✓ MICROPHONE audio track subscribed successfully!")
                # Set up audio frame listener for debugging
                setup_audio_listener(track, participant)
            else:
                print(f"🔊 Audio track subscribed (source: {publication.source})")
    
    @ctx.room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant
    ):
        print(f"📡 Track unsubscribed: {track.kind} from {participant.identity}")
    
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
    voice = "alloy"  # Default voice
    
    # Wait for participant to join and read their metadata
    try:
        print("⏳ Waiting for participant to join...")
        participant = await ctx.wait_for_participant()
        print(f"✓ Participant joined: {participant.identity}")
        
        # CRITICAL: Check for existing audio tracks and ensure they're subscribed
        print("🔍 Checking for existing audio tracks from participant...")
        audio_track_found = False
        for publication in participant.track_publications.values():
            print(f"  Found track publication: {publication.kind} (source: {publication.source}, subscribed: {publication.subscribed})")
            if publication.kind == rtc.TrackKind.KIND_AUDIO and publication.source == rtc.TrackSource.SOURCE_MICROPHONE:
                audio_track_found = True
                if not publication.subscribed:
                    print(f"  ⚠️  Audio track exists but not subscribed, subscribing now...")
                    try:
                        await publication.set_subscribed(True)
                        # Wait a moment for subscription to complete
                        await asyncio.sleep(0.2)
                        print(f"  ✓ Subscribed to audio track")
                        # Set up listener if track is now available
                        if publication.track:
                            setup_audio_listener(publication.track, participant)
                    except Exception as sub_error:
                        print(f"  ❌ Error subscribing to audio track: {sub_error}")
                elif publication.track:
                    print(f"  ✓ Audio track already subscribed, setting up listener")
                    setup_audio_listener(publication.track, participant)
        
        if not audio_track_found:
            print("  ⚠️  No microphone audio track found yet (user may publish it later)")
            print("  ℹ️  Waiting for user to publish audio track...")
            # Wait a bit for the user to publish their audio track
            max_wait_time = 5.0  # Wait up to 5 seconds
            wait_interval = 0.5
            waited = 0.0
            while waited < max_wait_time and not audio_track_found:
                await asyncio.sleep(wait_interval)
                waited += wait_interval
                # Check again for audio tracks
                for pub in participant.track_publications.values():
                    if pub.kind == rtc.TrackKind.KIND_AUDIO and pub.source == rtc.TrackSource.SOURCE_MICROPHONE:
                        audio_track_found = True
                        print(f"  ✓ Found audio track after waiting {waited:.1f}s")
                        if not pub.subscribed:
                            print(f"  ⚠️  Subscribing to audio track...")
                            await pub.set_subscribed(True)
                            await asyncio.sleep(0.2)
                        if pub.track:
                            setup_audio_listener(pub.track, participant)
                        break
                if audio_track_found:
                    break
            
            if not audio_track_found:
                print(f"  ⚠️  Still no audio track after {max_wait_time}s wait")
                print("  ℹ️  Audio track subscription will be handled by track_subscribed event handler")
        
        if participant.metadata:
            metadata = json.loads(participant.metadata)
            print(f"\n{'='*60}")
            print(f"📋 Participant metadata keys: {list(metadata.keys())}")
            print(f"{'='*60}\n")
            
            prompt = metadata.get("prompt", "")
            voice_from_metadata = metadata.get("voice", "")
            greeting_from_metadata = metadata.get("greetingInstructions", "")
            
            # Also try alternative key names in case of case sensitivity issues
            if not greeting_from_metadata:
                greeting_from_metadata = metadata.get("greeting_instructions", "")
            if not greeting_from_metadata:
                greeting_from_metadata = metadata.get("greetinginstructions", "")
            
            if voice_from_metadata:
                voice = voice_from_metadata
                print(f"✓ Voice set to: {voice}")
            
            if prompt:
                custom_instructions = prompt
                print(f"\n{'='*60}")
                print(f"✅ Prompt read from AccessToken metadata")
                print(f"{'='*60}")
                print(f"Prompt: {custom_instructions[:200]}...")
                print(f"{'='*60}\n")
            else:
                print("⚠️  No prompt found in participant metadata, using default instructions")
            
            if greeting_from_metadata:
                greeting_instructions = greeting_from_metadata
                print(f"\n{'='*60}")
                print(f"✅ Greeting instructions read from AccessToken metadata")
                print(f"{'='*60}")
                print(f"Greeting: {greeting_instructions}")
                print(f"{'='*60}\n")
            else:
                print(f"⚠️  No greeting instructions found in participant metadata")
                print(f"   Available keys: {list(metadata.keys())}")
                print(f"   Using default greeting: {greeting_instructions}")
        else:
            print("⚠️  No metadata found on participant, using default instructions")
    except Exception as e:
        print(f"⚠️  Error waiting for participant or reading metadata: {e}")
        print("Using default instructions")
    
    # Create session with the instructions from AccessToken metadata or default
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice=voice,
            model="gpt-realtime-mini"
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
                
                # Track LLM generation for agent responses
                if not is_user:
                    # Extract user ID from participant metadata
                    user_id = None
                    for p in ctx_ref.room.remote_participants.values():
                        if p.metadata:
                            try:
                                metadata = json.loads(p.metadata)
                                user_id = metadata.get("userId") or p.identity
                                break
                            except:
                                pass
                    
                    # Get recent user input from transcription history for context
                    recent_user_input = ""
                    if transcription_history:
                        # Get last user message
                        for item in reversed(transcription_history):
                            if item.get("role") == "user":
                                recent_user_input = item.get("message", "")
                                break
                    
                    # Track the generation
                    track_llm_generation(
                        model="gpt-4o-realtime-preview",  # OpenAI Realtime API model
                        input_text=recent_user_input,
                        output_text=text.strip(),
                        user_id=user_id,
                        room_name=ctx_ref.room.name,
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

    # CRITICAL: Before starting session, ensure we subscribe to all remote participant audio tracks
    print("\n🔍 Pre-session: Checking for remote participant audio tracks...")
    for remote_participant in ctx.room.remote_participants.values():
        print(f"  Checking participant: {remote_participant.identity}")
        for publication in remote_participant.track_publications.values():
            if publication.kind == rtc.TrackKind.KIND_AUDIO and publication.source == rtc.TrackSource.SOURCE_MICROPHONE:
                print(f"  ✓ Found microphone audio track: subscribed={publication.subscribed}, track={publication.track is not None}")
                if not publication.subscribed:
                    print(f"  ⚠️  Track not subscribed, subscribing now...")
                    try:
                        await publication.set_subscribed(True)
                        await asyncio.sleep(0.3)  # Wait for subscription to complete
                        print(f"  ✓ Successfully subscribed to audio track")
                        if publication.track:
                            setup_audio_listener(publication.track, remote_participant)
                    except Exception as sub_err:
                        print(f"  ❌ Error subscribing: {sub_err}")
                elif publication.track:
                    print(f"  ✓ Track already subscribed, setting up listener")
                    setup_audio_listener(publication.track, remote_participant)
    
    print("🚀 Starting agent session...")
    await session.start(
        room=ctx.room,
        agent=Assistant(instructions=custom_instructions, room=ctx.room),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
            # Enable video input for multimodal capabilities
            video_input=True,
        ),
    )
    print("✓ Agent session started")
    
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

    # Debug: Print the greeting instructions that will be used
    print(f"\n{'='*60}")
    print(f"🎤 About to generate greeting with instructions:")
    print(f"{'='*60}")
    print(f"{greeting_instructions}")
    print(f"{'='*60}\n")
    
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