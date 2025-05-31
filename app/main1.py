# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager
from sqlmodel import Field, Session, SQLModel, create_engine, select
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import base64
import hashlib
import os
import secrets
import logging
from datetime import datetime, timedelta



# Set up logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger('apscheduler').setLevel(logging.WARNING)

# --- Configuration ---
DATABASE_URL = "sqlite:///./database.db"
# Make sure to set these environment variables securely
# SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
# SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_CLIENT_ID='faa0134745184b2094651b9c44c1f67e'
SPOTIFY_CLIENT_SECRET="03be1bd5caa04defa182786bc4a40918"
print("Spotify Client ID:", SPOTIFY_CLIENT_ID)  # Debugging line, remove in production
# This redirect URI must match what's configured in your Spotify Developer Dashboard
SPOTIFY_REDIRECT_URI = "https://127.0.0.1:8000/callback"
# Path to store generated audio files
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

# --- Database Models (SQLModel) ---
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    spotify_id: str = Field(unique=True, index=True)
    access_token: str
    refresh_token: str
    token_expires_at: datetime
    scope: str

class Track(SQLModel, table=True):
    # This will store Spotify track details if you need to cache them
    id: str = Field(primary_key=True)
    name: str
    artist: str
    duration_ms: int
    # Add other fields from Spotify's Track object as needed

class RadioItem(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id") # Link to a specific user
    text: str
    before_track_id: str # Spotify ID of the track before the radio item
    after_track_id: str # Spotify ID of the track after the radio item
    audio_file_path: str | None = Field(default=None) # Path to the generated audio file
    generated_at: datetime = Field(default_factory=datetime.now)

engine = create_engine(DATABASE_URL)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# --- Global State Management (for a single user, or per-user if multi-user) ---
# For a multi-user setup, these would be stored per-user in the database or a cache
# For simplicity, initially, let's assume one active user for the polling
active_user_spotify_id: str | None = None
active_user_session: Session | None = None # To hold the session for the active user's data

current_song: Dict[str, Any] | None = None # Represents Spotify's 'currently playing' object
current_queue: List[Dict[str, Any]] = [] # List of Spotify Track objects for the user's queue

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected: {websocket.client}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                self.disconnect(connection)
            except Exception as e:
                logger.error(f"Error broadcasting message to {connection.client}: {e}")

manager = ConnectionManager()

# --- Spotify API Utility ---
def get_spotify_client(user_id: str) -> spotipy.Spotify | None:
    with Session(engine) as session:
        user = session.exec(select(User).where(User.spotify_id == user_id)).first()
        if not user:
            logger.warning(f"No user found for spotify_id: {user_id}")
            return None

        # Check if token needs refreshing
        if user.token_expires_at < datetime.now() + timedelta(minutes=5): # Refresh if expiring soon
            logger.info(f"Access token for {user.spotify_id} is expiring, attempting refresh...")
            auth_manager = SpotifyOAuth(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                redirect_uri=SPOTIFY_REDIRECT_URI,
                scope=user.scope # Use the original scope
            )
            try:
                # The refresh token itself is handled by SpotifyOAuth
                # We essentially simulate a token grant from a refresh token
                token_info = auth_manager.refresh_access_token(user.refresh_token)
                user.access_token = token_info['access_token']
                user.refresh_token = token_info.get('refresh_token', user.refresh_token) # Refresh token might change
                user.token_expires_at = datetime.now() + timedelta(seconds=token_info['expires_in'])
                session.add(user)
                session.commit()
                session.refresh(user)
                logger.info(f"Access token for {user.spotify_id} refreshed successfully.")
            except Exception as e:
                logger.error(f"Failed to refresh token for {user.spotify_id}: {e}")
                return None
        
        sp = spotipy.Spotify(auth=user.access_token)
        return sp

# --- Spotify API Calls (Mimicking your network.ts) ---
async def fetch_playback_state_spotify(spotify_id: str) -> Dict[str, Any] | None:
    sp = get_spotify_client(spotify_id)
    if not sp:
        return None
    try:
        playback_state = sp.current_playback()
        return playback_state
    except Exception as e:
        logger.error(f"Error fetching playback state for {spotify_id}: {e}")
        return None

async def fetch_user_queue_spotify(spotify_id: str) -> List[Dict[str, Any]]:
    sp = get_spotify_client(spotify_id)
    if not sp:
        return []
    try:
        queue_data = sp.queue()
        # Spotify's queue API response is a bit complex, extract tracks
        tracks = []
        if queue_data and queue_data.get('queue'):
            for item in queue_data['queue']:
                if item['type'] == 'track':
                    tracks.append(item)
        if queue_data and queue_data.get('currently_playing') and queue_data['currently_playing']['type'] == 'track':
            # Add currently playing to the front of the queue if not already there
            if not tracks or tracks[0]['id'] != queue_data['currently_playing']['id']:
                tracks.insert(0, queue_data['currently_playing'])
        return tracks
    except Exception as e:
        logger.error(f"Error fetching user queue for {spotify_id}: {e}")
        return []

async def pause_song_spotify(spotify_id: str):
    sp = get_spotify_client(spotify_id)
    if sp:
        try:
            sp.pause_playback()
            logger.info(f"Song paused for {spotify_id}")
        except Exception as e:
            logger.error(f"Error pausing song for {spotify_id}: {e}")

async def resume_song_spotify(spotify_id: str):
    sp = get_spotify_client(spotify_id)
    if sp:
        try:
            sp.start_playback()
            logger.info(f"Song resumed for {spotify_id}")
        except Exception as e:
            logger.error(f"Error resuming song for {spotify_id}: {e}")

# --- AI/Audio Generation (Placeholder - Replace with your actual logic) ---
async def generate_queue_audio(text_to_speak: str) -> str | None:
    """
    Placeholder for your audio generation logic.
    Returns file path to generated audio (e.g., WAV).
    """
    logger.info(f"Generating audio for text: '{text_to_speak}'...")
    # In a real scenario, this would call an external TTS API or a local model.
    # For demonstration, let's create a dummy file.
    dummy_audio_file = os.path.join(AUDIO_DIR, f"radio_item_{secrets.token_hex(8)}.wav")
    try:
        # Simulate writing a small dummy WAV file
        with open(dummy_audio_file, "wb") as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
        logger.info(f"Dummy audio generated at: {dummy_audio_file}")
        return dummy_audio_file
    except Exception as e:
        logger.error(f"Error creating dummy audio file: {e}")
        return None

async def generate_queue_texts(tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Placeholder for your text generation logic.
    Generates a text description for radio item based on tracks.
    """
    logger.info(f"Generating text for tracks: {tracks}")
    if len(tracks) >= 2:
        text = f"Up next, after {tracks[0]['name']} by {tracks[0]['artists'][0]['name']}, we'll hear {tracks[1]['name']} by {tracks[1]['artists'][0]['name']}."
        return {
            "text": text,
            "before_track_id": tracks[1]['id'],
            "after_track_id": tracks[0]['id'],
            "audio_file_path": None # Will be generated later
        }
    elif len(tracks) == 1:
        text = f"You're currently listening to {tracks[0]['name']} by {tracks[0]['artists'][0]['name']}."
        return {
            "text": text,
            "before_track_id": tracks[0]['id'], # This might need adjustment based on how you use it
            "after_track_id": None,
            "audio_file_path": None
        }
    return {"text": "Enjoy the music!", "before_track_id": None, "after_track_id": None, "audio_file_path": None}


# --- Background Polling Task (Replaces Foreground Service) ---
async def update_spotify_status():
    """
    This function mimics your updateSongStatus and runs periodically.
    """
    global current_song, current_queue, active_user_spotify_id

    # if not active_user_spotify_id:
    #     logger.warning("No active user set for polling. Skipping update.")
    #     return

    #logger.info("Running update_spotify_status task...")
    try:
        currently_playing = await fetch_playback_state_spotify(active_user_spotify_id)

        if not currently_playing or not currently_playing.get('item'):
            #logger.info('No playback state found or nothing playing. Stopping polling.')
            # Potentially stop scheduler here if no user is active or playing for too long
            await manager.broadcast({"type": "error", "message": "No song currently playing on Spotify. Polling paused."})
            return

        new_current_song = currently_playing['item']
        new_queue = await fetch_user_queue_spotify(active_user_spotify_id)

        # Check for song change
        if not current_song or new_current_song['id'] != current_song.get('id'):
            logger.info(f"Song changed! New song: {new_current_song['name']}")
            current_song = new_current_song
            current_queue = new_queue

            await manager.broadcast({"type": "currentSong", "data": current_song})
            await manager.broadcast({"type": "updatedQueue", "data": current_queue})

            # Check for radio items to play
            with Session(engine) as session:
                user = session.exec(select(User).where(User.spotify_id == active_user_spotify_id)).first()
                if user:
                    # Fetch and filter radio items for the active user
                    radio_items = session.exec(
                        select(RadioItem)
                        .where(RadioItem.user_id == user.id)
                        .order_by(RadioItem.generated_at)
                    ).all()

                    # Filter out invalid radio items (e.g., if tracks are no longer in queue)
                    valid_radio_items = []
                    for item in radio_items:
                        after_track_exists = any(t.get('id') == item.after_track_id for t in current_queue)
                        before_track_exists = any(t.get('id') == item.before_track_id for t in current_queue)
                        # Check if the radio item is still relevant to the current queue context
                        # This logic needs to be robust, similar to your original loadAndFilterRadioItems
                        if after_track_exists and before_track_exists:
                            valid_radio_items.append(item)
                        else:
                            # Delete orphaned audio file if it exists
                            if item.audio_file_path and os.path.exists(item.audio_file_path):
                                os.remove(item.audio_file_path)
                                logger.info(f"Deleted orphaned audio file: {item.audio_file_path}")
                            session.delete(item)
                            session.commit() # Commit deletion
                    
                    # Update radio items in DB after filtering
                    # This requires re-adding and flushing, or just letting the session update if objects are tracked
                    session.add_all(valid_radio_items) # Re-add to ensure they are tracked
                    session.commit()
                    session.refresh(user) # Refresh user to get updated radio_items if needed

                    # Sort radio items by relevance to current song/queue
                    # Find the first radio item that matches the current song's context
                    first_radio_item_to_play = None
                    if current_song and current_queue:
                        # Logic: if current song is after_track_id, and next in queue is before_track_id
                        current_song_index_in_queue = -1
                        for i, track in enumerate(current_queue):
                            if track.get('id') == current_song.get('id'):
                                current_song_index_in_queue = i
                                break
                        
                        if current_song_index_in_queue != -1 and current_song_index_in_queue + 1 < len(current_queue):
                            next_track_id_in_queue = current_queue[current_song_index_in_queue + 1].get('id')
                            
                            for item in valid_radio_items:
                                if item.after_track_id == current_song['id'] and item.before_track_id == next_track_id_in_queue:
                                    first_radio_item_to_play = item
                                    break
                                
                    if first_radio_item_to_play and first_radio_item_to_play.audio_file_path:
                        logger.info(f"Playing radio item before {first_radio_item_to_play.before_track_id}")
                        await pause_song_spotify(active_user_spotify_id)
                        # Broadcast audio file path to frontend, frontend will play it
                        await manager.broadcast({
                            "type": "playRadioAudio",
                            "audioUrl": f"/audio/{os.path.basename(first_radio_item_to_play.audio_file_path)}"
                        })
                        # Frontend should send a confirmation when audio finishes
                        # For now, simulate delay and then resume
                        # In a real app, the frontend would tell the backend when audio finishes
                        await asyncio.sleep(5) # Simulate audio playback duration
                        await resume_song_spotify(active_user_spotify_id)
                        
                        # Remove played radio item and delete its audio file
                        if os.path.exists(first_radio_item_to_play.audio_file_path):
                            os.remove(first_radio_item_to_play.audio_file_path)
                            logger.info(f"Deleted played audio file: {first_radio_item_to_play.audio_file_path}")
                        session.delete(first_radio_item_to_play)
                        session.commit()
                        logger.info(f"Radio item {first_radio_item_to_play.id} played and removed.")
                    else:
                        logger.info("No relevant radio item to play or audio not generated yet.")

                    # Generate new radio items if needed
                    # Logic similar to your fetchAudioTexts
                    if current_queue and len(current_queue) >= 2:
                        existing_texts_for_next_pair = session.exec(
                            select(RadioItem)
                            .where(RadioItem.user_id == user.id)
                            .where(RadioItem.after_track_id == current_queue[0]['id'])
                            .where(RadioItem.before_track_id == current_queue[1]['id'])
                        ).first()

                        if not existing_texts_for_next_pair:
                            logger.info("Generating new radio item text for upcoming tracks...")
                            new_radio_text_data = await generate_queue_texts([current_queue[0], current_queue[1]])
                            
                            if new_radio_text_data and new_radio_text_data.get("text"):
                                new_radio_item = RadioItem(
                                    user_id=user.id,
                                    text=new_radio_text_data["text"],
                                    before_track_id=new_radio_text_data["before_track_id"],
                                    after_track_id=new_radio_text_data["after_track_id"],
                                    audio_file_path=None # Audio to be generated asynchronously
                                )
                                session.add(new_radio_item)
                                session.commit()
                                session.refresh(new_radio_item)
                                logger.info(f"New radio item text generated: {new_radio_item.text}")
                                # Asynchronously generate audio
                                audio_path = await generate_queue_audio(new_radio_item.text)
                                if audio_path:
                                    new_radio_item.audio_file_path = audio_path
                                    session.add(new_radio_item)
                                    session.commit()
                                    session.refresh(new_radio_item)
                                    logger.info(f"Audio generated and saved for radio item {new_radio_item.id}")
                                # Broadcast updated radio items to frontend
                                await manager.broadcast({"type": "updatedRadioItems", "data": new_radio_item.model_dump()}) # Send the new item


        # Check if the song is about to end (within 5 seconds)
        if currently_playing.get('is_playing') and currently_playing.get('item') and currently_playing.get('progress_ms') is not None:
            remaining_time = currently_playing['item']['duration_ms'] - currently_playing['progress_ms']
            if remaining_time < 5000: # Less than 5 seconds
                logger.info(f"Song '{currently_playing['item']['name']}' is about to finish in {remaining_time} ms.")
                # You might want to pre-fetch the next queue items or radio items here if you haven't already.
                # Or simply wait for the next polling cycle, which will detect the song change.

    except Exception as e:
        logger.error(f"Error in update_spotify_status: {e}")
        await manager.broadcast({"type": "error", "message": f"Server error during Spotify update: {e}"})

# --- FastAPI App Lifecycle (Lifespan) ---
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application...")
    create_db_and_tables() # Create database tables on startup
    
    # Start the scheduler
    scheduler.add_job(update_spotify_status, 'interval', seconds=5) # Poll every 5 seconds
    scheduler.start()
    logger.info("Scheduler started.")
    
    yield # Application runs
    
    logger.info("Shutting down FastAPI application...")
    scheduler.shutdown(wait=True) # Shut down scheduler gracefully
    logger.info("Scheduler stopped.")

app = FastAPI(lifespan=lifespan)

# Mount static files directory to serve audio
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# --- HTML for WebSocket test (optional, for quick testing) ---
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>FastAPI WebSocket</title>
        <script>
            var ws = null;
            var spotifyIdInput = null; // To hold the user's Spotify ID for authentication
            var currentSongDiv = null;
            var queueDiv = null;
            var radioItemsDiv = null;
            var spotifyAuthLink = null;
            var serverMessageDiv = null;

            window.onload = function() {
                currentSongDiv = document.getElementById('currentSong');
                queueDiv = document.getElementById('queue');
                radioItemsDiv = document.getElementById('radioItems');
                spotifyIdInput = document.getElementById('spotifyIdInput');
                spotifyAuthLink = document.getElementById('spotifyAuthLink');
                serverMessageDiv = document.getElementById('serverMessage');

                // Generate a random code verifier for PKCE
                function generateRandomString(length) {
                    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
                    let text = '';
                    for (let i = 0; i < length; i++) {
                        text += possible.charAt(Math.floor(Math.random() * possible.length));
                    }
                    return text;
                }

                function sha256(plain) {
                    const encoder = new TextEncoder();
                    const data = encoder.encode(plain);
                    return window.crypto.subtle.digest('SHA-256', data);
                }

                function base64urlencode(input) {
                    return btoa(String.fromCharCode(...new Uint8Array(input)))
                        .replace(/=/g, '')
                        .replace(/\+/g, '-')
                        .replace(/\//g, '_');
                }

                async function generateCodeChallenge(codeVerifier) {
                    const hashed = await sha256(codeVerifier);
                    return base64urlencode(hashed);
                }

                spotifyAuthLink.onclick = async function(event) {
                    event.preventDefault();
                    const clientId = 'faa0134745184b2094651b9c44c1f67e'; // Replace with your actual client ID
                    const redirectUri = 'https://127.0.0.1:8000/callback';
                    const scope = 'user-read-playback-state user-read-queue user-modify-playback-state';

                    const codeVerifier = generateRandomString(128);
                    const codeChallenge = await generateCodeChallenge(codeVerifier);

                    // Store code_verifier temporarily (e.g., in localStorage)
                    localStorage.setItem('spotify_code_verifier', codeVerifier);

                    const authUrl = `https://accounts.spotify.com/authorize?` +
                                    `response_type=code` +
                                    `&client_id=${clientId}` +
                                    `&scope=user-read-private%20user-read-email%20user-read-currently-playing%20user-read-playback-state%20user-modify-playback-state` +
                                    `&redirect_uri=${encodeURIComponent(redirectUri)}` +
                                    `&code_challenge=${codeChallenge}` +
                                    `&code_challenge_method=S256`;
                    window.location.href = authUrl;
                };

                // WebSocket connection
                function connectWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                    
                    ws.onopen = function(event) {
                        console.log("WebSocket opened:", event);
                        serverMessageDiv.textContent = "Connected to server.";
                    };

                    ws.onmessage = function(event) {
                        const message = JSON.parse(event.data);
                        console.log("Received message:", message);
                        serverMessageDiv.textContent = `Last server message: ${message.type}`;

                        if (message.type === 'currentSong') {
                            currentSongDiv.innerHTML = `<h3>Current Song:</h3><p>${message.data.name} by ${message.data.artists[0].name}</p>`;
                        } else if (message.type === 'updatedQueue') {
                            let queueHtml = '<h3>Queue:</h3><ul>';
                            message.data.forEach(track => {
                                queueHtml += `<li>${track.name} by ${track.artists[0].name}</li>`;
                            });
                            queueHtml += '</ul>';
                            queueDiv.innerHTML = queueHtml;
                        } else if (message.type === 'updatedRadioItems') {
                            // This might be a single new item, or a full list
                            radioItemsDiv.innerHTML = `<h3>Radio Items:</h3><p>New item generated: "${message.data.text}"</p>`;
                            if (message.data.audio_file_path) {
                                radioItemsDiv.innerHTML += `<audio controls src="/audio/${message.data.audio_file_path.split('/').pop()}"></audio>`;
                            }
                        } else if (message.type === 'playRadioAudio') {
                            const audio = new Audio(message.audioUrl);
                            audio.play();
                            audio.onended = () => {
                                console.log("Radio audio finished.");
                                // Optionally, send a message back to backend that audio finished
                                // ws.send(JSON.stringify({ type: 'radioAudioFinished' }));
                            };
                            serverMessageDiv.textContent = `Playing radio audio: ${message.audioUrl}`;
                        } else if (message.type === 'error') {
                            serverMessageDiv.textContent = `ERROR: ${message.message}`;
                            console.error("Server Error:", message.message);
                        }
                    };

                    ws.onclose = function(event) {
                        console.log("WebSocket closed:", event);
                        serverMessageDiv.textContent = `Disconnected. Code: ${event.code}, Reason: ${event.reason}. Reconnecting in 5s...`;
                        setTimeout(connectWebSocket, 5000); // Attempt to reconnect
                    };

                    ws.onerror = function(event) {
                        console.error("WebSocket error:", event);
                        serverMessageDiv.textContent = `WebSocket Error! See console.`;
                    };
                }

                // Initial WebSocket connection
                connectWebSocket();
            };

            function selectUser() {
                const spotifyId = spotifyIdInput.value;
                if (ws && spotifyId) {
                    ws.send(JSON.stringify({ type: 'selectUser', spotify_id: spotifyId }));
                    serverMessageDiv.textContent = `Attempting to select user: ${spotifyId}`;
                } else {
                    serverMessageDiv.textContent = "Enter Spotify ID and ensure WebSocket is connected.";
                }
            }
        </script>
    </head>
    <body>
        <h1>FastAPI Spotify Integration</h1>
        <p><a href="#" id="spotifyAuthLink">Login with Spotify</a></p>
        <p>Enter Spotify User ID to activate polling:</p>
        <input type="text" id="spotifyIdInput" placeholder="Spotify User ID (e.g., your_username)">
        <button onclick="selectUser()">Activate Polling for User</button>
        <hr/>
        <div id="serverMessage" style="color: blue;">Connecting...</div>
        <div id="currentSong"></div>
        <div id="queue"></div>
        <div id="radioItems"></div>
    </body>
</html>
"""

# --- FastAPI Endpoints ---
@app.get("/")
async def get():
    return HTMLResponse(html)

@app.get("/callback")
async def spotify_callback(code: str, state: str | None = None):
    print("Spotify callback received with code:", code)
    # This endpoint receives the authorization code from Spotify
    # It must be the same as your SPOTIFY_REDIRECT_URI
    
    # Retrieve code_verifier from a temporary storage (e.g., database or session)
    # For this example, we'll assume it's passed via some client-side mechanism
    # In a real app, you'd associate 'state' with a user session to retrieve the correct verifier
    
    # IMPORTANT: In a real-world scenario, the code_verifier should be passed
    # from the frontend to the backend securely (e.g., in the body of a POST request
    # after the redirect, or via a temporary server-side session that the frontend can query).
    # For this simplified example, we're assuming the frontend would somehow get it to the backend.
    
    # For a full PKCE flow, the frontend sends the code_verifier
    # along with the authorization code.
    # Here, we'll simulate the backend generating it for a demonstration,
    # but normally the frontend generates and passes it.
    
    # A more robust solution involves a temporary storage on the server
    # linked to the 'state' parameter or the user's session.
    logger.info(f"Received Spotify callback with code: {code}")

    # Simulating code_verifier retrieval for demonstration
    # In a real app, this would be from the frontend's request or your session store
    # This is not how you'd normally handle PKCE backend;
    # the frontend passes the verifier back to the backend.
    # For a demo, let's just make one up, assuming the frontend provided it.
    # For a real implementation, you'd need the *original* code_verifier
    # that was used to generate the code_challenge sent to Spotify.
    
    # A better approach for testing:
    # 1. Frontend sends verifier to backend at a /start_auth endpoint.
    # 2. Backend stores verifier in DB/cache, redirects to Spotify with state.
    # 3. Spotify redirects back to /callback. Backend retrieves verifier using state.
    # For this direct callback, assume frontend handles verifier transmission.
    
    # For the purpose of this example, let's create a dummy flow:
    # We will need the actual code_verifier generated by the frontend.
    # Let's assume the frontend sends the code_verifier in a subsequent request to /exchange_token
    # after receiving the code, or through a cookie/session.
    
    # For a basic example of token exchange (without full PKCE backend handling here,
    # as the frontend holds the verifier), we'll use SpotifyOAuth.
    # If using PKCE fully, you'd manually perform the POST request with the verifier.
    
    # Let's use SpotifyOAuth's built-in exchange if possible for simplicity in demo
    # It usually handles the PKCE logic implicitly if `redirect_uri` is set up.
    
    auth_manager = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope="user-read-playback-state user-read-queue user-modify-playback-state", # Add all needed scopes
        cache_path=None # We'll manage tokens in DB
    )
    
    try:
        token_info = auth_manager.get_access_token(code=code, as_dict=True)
        
        # Get user info to store spotify_id
        sp = spotipy.Spotify(auth=token_info['access_token'])
        user_profile = sp.current_user()
        spotify_id = user_profile['id']

        with Session(engine) as session:
            user = session.exec(select(User).where(User.spotify_id == spotify_id)).first()
            if not user:
                user = User(
                    spotify_id=spotify_id,
                    access_token=token_info['access_token'],
                    refresh_token=token_info.get('refresh_token', ''),
                    token_expires_at=datetime.now() + timedelta(seconds=token_info['expires_in']),
                    scope=token_info['scope']
                )
            else:
                user.access_token = token_info['access_token']
                user.refresh_token = token_info.get('refresh_token', user.refresh_token)
                user.token_expires_at = datetime.now() + timedelta(seconds=token_info['expires_in'])
                user.scope = token_info['scope']
            
            session.add(user)
            session.commit()
            session.refresh(user)
            logger.info(f"User {spotify_id} tokens saved/updated.")

        return HTMLResponse(f"<h1>Spotify Authorization Successful!</h1><p>You can now close this tab and return to the application.</p><p>Your Spotify ID: <strong>{spotify_id}</strong> (Use this to activate polling)</p>")

    except Exception as e:
        logger.error(f"Error during token exchange: {e}")
        return HTMLResponse(f"<h1>Spotify Authorization Failed</h1><p>Error: {e}</p>", status_code=500)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            logger.info(f"Received WebSocket message: {data}")
            if data.get("type") == "selectUser":
                spotify_id = data.get("spotify_id")
                if spotify_id:
                    global active_user_spotify_id
                    active_user_spotify_id = spotify_id
                    # Initial fetch on user selection
                    await manager.send_personal_message(f"Polling activated for Spotify user: {spotify_id}", websocket)
                    await update_spotify_status() # Trigger an immediate update
                else:
                    await manager.send_personal_message("Spotify ID is required to select user.", websocket)
            # Add other WebSocket message handlers here if your frontend sends commands
            # For example, 'playAudioFinished' to tell backend when radio audio is done
            # or 'clearData' to trigger clearAllAppData
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.send_personal_message(f"An error occurred: {e}", websocket)

# Example of how you'd trigger clearAllAppData from an API endpoint or WebSocket
@app.post("/clear_all_app_data")
async def clear_data_endpoint():
    """
    Clears all user data and audio files.
    This should be protected by authentication in a real application.
    """
    try:
        logger.info('[Cleanup] Starting data cleanup...')
        with Session(engine) as session:
            # Delete all users and their associated radio items
            session.exec("DELETE FROM RadioItem")
            session.exec("DELETE FROM User")
            session.commit()
            logger.info('[Cleanup] Database cleared successfully')

        # Clear document directory (where we store audio files)
        logger.info('[Cleanup] Clearing audio files directory...')
        for filename in os.listdir(AUDIO_DIR):
            file_path = os.path.join(AUDIO_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f'[Cleanup] Failed to delete {file_path}. Reason: {e}')
        logger.info('[Cleanup] Audio files cleared successfully')
        
        global active_user_spotify_id, current_song, current_queue
        active_user_spotify_id = None
        current_song = None
        current_queue = []

        await manager.broadcast({"type": "message", "message": "All app data cleared."})
        return {"status": "success", "message": "All app data cleared successfully."}
    except Exception as e:
        logger.error(f'[Cleanup] Error during data cleanup: {e}')
        raise HTTPException(status_code=500, detail=f"Failed to clear app data: {e}")

# To run this:
# 1. pip install fastapi uvicorn websockets apscheduler sqlmodel spotipy
# 2. Set environment variables:
#    export SPOTIFY_CLIENT_ID="your_spotify_client_id"
#    export SPOTIFY_CLIENT_SECRET="your_spotify_client_secret"
# 3. uvicorn main:app --reload