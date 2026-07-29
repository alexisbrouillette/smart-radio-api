import asyncio
from fastapi import FastAPI, Body, HTTPException, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import httpx
from pydantic import BaseModel
from typing import List
from functions.classes import Track
from functions.audio_gen import get_audio
from functions.text_gen import generate_text_for_song, get_llm
import threading
import queue
from functools import wraps
import os
import re

# Manual parser for .env file to load GEMINI_API_KEY without requiring python-dotenv
if os.path.exists(".env"):
    try:
        with open(".env") as f:
            for line in f:
                clean_line = line.strip()
                if clean_line and not clean_line.startswith("#") and "=" in clean_line:
                    key, val = clean_line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")
        print("Loaded environment variables from local .env file")
    except Exception as e:
        print(f"Warning: could not parse local .env file: {e}")
import gc
from concurrent.futures import ThreadPoolExecutor
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow requests from this origin
    allow_credentials=True,
    allow_methods=["*"],  # allow all methods
    allow_headers=["*"],  # allow all headers
)

llm = get_llm()

# Create a queue for TTS tasks and a single-threaded executor
app.gpu_queue = asyncio.Queue()
app.gpu_executor = ThreadPoolExecutor(max_workers=1)  # Process one TTS request at a time to keep CPU usage controlled

TTS_CACHE_DIR = "/tmp/smart_radio_tts_cache"
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

# Tracks which TTS texts are currently being synthesized to avoid duplicate work
_tts_in_progress: dict = {}

# pending_speech[trackKey] = hostText: speech to inject before a specific track
# Frontend schedules this when a radio item is generated; server injects it at stream time
pending_speech: dict = {}

def normalize_key(s: str) -> str:
    if not s:
        return ""
    import re
    return re.sub(r'[^\w]', '', s.lower())

def get_tts_cache_path(text: str) -> str:
    import hashlib
    h = hashlib.md5(text.strip().lower().encode('utf-8')).hexdigest()
    return os.path.join(TTS_CACHE_DIR, f"tts_{h}.mp3")

class GPUTask:
    def __init__(self, text_for_audio: str):
        self.text_for_audio = text_for_audio
        self.task_id = str(uuid.uuid4())
        # Use stable cache path based on text content
        self.result_file = get_tts_cache_path(text_for_audio)
        self.completion_event = asyncio.Event()
        self.error = None

async def gpu_worker():
    """Worker that processes TTS tasks one at a time"""
    print("Starting TTS Worker")
    while True:
        task = await app.gpu_queue.get()
        print(f"Processing TTS task: {task.task_id} (queue size: {app.gpu_queue.qsize()})")
        
        # Skip synthesis if already cached on disk
        if os.path.exists(task.result_file) and os.path.getsize(task.result_file) > 1000:
            print(f"[TTS CACHE HIT] Reusing cached TTS for task {task.task_id}")
            task.completion_event.set()
            app.gpu_queue.task_done()
            # Clean up in-progress tracker
            _tts_in_progress.pop(task.result_file, None)
            continue
        
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                app.gpu_executor, 
                execute_gpu_task, 
                task.text_for_audio, 
                task.result_file
            )
            print(f"TTS task {task.task_id} completed successfully")
        except Exception as e:
            print(f"TTS task {task.task_id} failed: {e}")
            task.error = e
        finally:
            task.completion_event.set()
            app.gpu_queue.task_done()
            _tts_in_progress.pop(task.result_file, None)

def execute_gpu_task(text_for_audio: str, output_file: str):
    """Synchronous function that runs the actual Kokoro CPU operation and converts to MP3"""
    try:
        from pydub import AudioSegment
        import static_ffmpeg
        ffmpeg_bin, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
        AudioSegment.converter = ffmpeg_bin

        print(f"Executing Kokoro TTS task for text: {text_for_audio[:50]}...")
        
        # Parse language prefix
        lang = 'en'
        text = text_for_audio
        if text_for_audio.startswith("fr:"):
            lang = 'fr'
            text = text_for_audio[3:].strip()
        elif text_for_audio.startswith("en:"):
            lang = 'en'
            text = text_for_audio[3:].strip()
            
        wav_temp = output_file.replace('.mp3', '.wav')
        get_audio(text, wav_temp, lang=lang)
        
        # Convert WAV to broadcast-compliant 44.1kHz stereo MP3
        if os.path.exists(wav_temp):
            sound = AudioSegment.from_wav(wav_temp)
            sound = sound.set_frame_rate(44100).set_channels(2)
            sound.export(output_file, format="mp3", bitrate="128k")
            os.remove(wav_temp)
            print(f"[TTS CONVERT SUCCESS] Saved 44.1kHz stereo MP3 DJ Speech -> {output_file} ({os.path.getsize(output_file)} bytes)")
        
        gc.collect()
        return output_file
    except Exception as e:
        print(f"Error in TTS task execution: {e}")
        raise

@app.on_event("startup")
async def start_workers():
    asyncio.create_task(gpu_worker())

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/test")
def read_test():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Background Audio Test</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: system-ui, -apple-system, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
                background: #121212;
                color: white;
                text-align: center;
            }
            button {
                padding: 15px 30px;
                font-size: 18px;
                background: #1DB954;
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-weight: bold;
                margin-bottom: 20px;
            }
            #log {
                font-family: monospace;
                background: #222;
                padding: 15px;
                border-radius: 8px;
                width: 80%;
                max-width: 400px;
                height: 150px;
                overflow-y: auto;
                text-align: left;
            }
        </style>
    </head>
    <body>
        <h1>Background JS Test</h1>
        <p>Tap start, lock your screen, and check if you hear a beep every 5 seconds.</p>
        <button id="btn" onclick="startTest()">Start Test</button>
        <div id="log">Logs:</div>

        <script>
            let audioCtx;
            let bgAudio;
            let count = 0;

            const silentWav = "data:audio/wav;base64,UklGRjIAAABXQVZFZm10IBIAAAABAAEAQB8AAEAfAAABAAgAAABmYWN0BAAAAAAAAABkYXRhAAAAAA==";

            function log(msg) {
                const el = document.getElementById('log');
                el.innerHTML += '<br>' + new Date().toLocaleTimeString() + ': ' + msg;
                el.scrollTop = el.scrollHeight;
            }

            function playBeep() {
                try {
                    if (!audioCtx) return;
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(600, audioCtx.currentTime); // 600Hz beep
                    
                    gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.3);
                    
                    osc.connect(gain);
                    gain.connect(audioCtx.destination);
                    
                    osc.start();
                    osc.stop(audioCtx.currentTime + 0.3);
                    count++;
                    log("Beep #" + count);
                } catch (e) {
                    log("Beep error: " + e.message);
                }
            }

            function startTest() {
                document.getElementById('btn').disabled = true;
                log("Initializing HTML5 looped audio...");
                
                // Set up continuous silent audio player
                bgAudio = new Audio(silentWav);
                bgAudio.loop = true;
                
                // Setup Media Session to register as a background player with the OS
                if ('mediaSession' in navigator) {
                    navigator.mediaSession.metadata = new MediaMetadata({
                        title: 'Background Engine Active',
                        artist: 'Smart Radio',
                        album: 'Keep-Alive Service'
                    });
                }
                
                bgAudio.play()
                    .then(() => {
                        log("Looped silent audio started playing.");
                        
                        log("Initializing Web AudioContext...");
                        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                        
                        // Schedule beeps
                        playBeep();
                        setInterval(playBeep, 5000);
                    })
                    .catch(e => {
                        log("Failed to start audio loop: " + e.message);
                        document.getElementById('btn').disabled = false;
                    });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/radio")
def generate_radio():
    return {"Hello": "World111"}

@app.post("/get_radio_text")
def get_queue_radio(queue: List[Track]):
    print("coucou!")
    try:
        songs_texts = generate_text_for_song(queue, llm)
        if not songs_texts or not songs_texts.strip():
            raise ValueError("Model returned an empty transition script.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in generate_text_for_song: {e}. Falling back to default script.")
        prev_song = queue[0].name if len(queue) > 0 else "that track"
        prev_artist = queue[0].artists if len(queue) > 0 else "the artist"
        next_song = queue[-1].name if len(queue) > 1 else "the next track"
        next_artist = queue[-1].artists if len(queue) > 1 else "our next artist"
        songs_texts = f"That was {prev_song} by {prev_artist}. Up next, here is {next_song} by {next_artist}!"
    print("Generated text: ", songs_texts)

    lastTrack = queue[-1]
    radio = {
        "beforeTrackId": lastTrack.id,
        "afterTrackId": queue[0].id,
        "text": songs_texts,
        "audio": "empty",
    }

    return radio

@app.post("/get_radio_audio")
async def generate_audio_from_text(textForAudio: str = Body(...)):
    """Generate audio using CPU queue with proper async coordination"""
    try:
        # Create a task with completion event
        task = GPUTask(textForAudio)
        
        # Add to queue
        await app.gpu_queue.put(task)
        print(f"Queued TTS task: {task.task_id}")
        
        # Wait for completion event with timeout (5 minutes)
        try:
            await asyncio.wait_for(task.completion_event.wait(), timeout=300.0)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Audio generation timed out")
        
        # Check for errors
        if task.error:
            raise HTTPException(status_code=500, detail=f"Audio generation failed: {task.error}")
        
        # Return the generated file
        if os.path.exists(task.result_file):
            headers = {'Content-Disposition': f'attachment; filename="{task.result_file}"'}
            
            # Note: The file is retained for delivery, but can be cleaned up later or left as is.
            return FileResponse(task.result_file, headers=headers, media_type="audio/wav")
        else:
            raise HTTPException(status_code=500, detail="Generated audio file not found")
        
    except HTTPException:
        raise
    except Exception as e:
        print("Error generating audio: ", e)
        raise HTTPException(status_code=500, detail="Error generating audio")

@app.get("/spotify/queue")
async def get_spotify_queue(authorization: str = Header(...)):
    SPOTIFY_API_URL = "https://api.spotify.com/v1/me/player/queue"
    try:
        async with httpx.AsyncClient() as client:
            # Forward the request to Spotify
            response = await client.get(
                SPOTIFY_API_URL,
                headers={"Authorization": authorization},
            )
            response.raise_for_status()  # Raise errors for 4xx/5xx
            queue_data = response.json()
            # Remove duplicates while preserving order
            seen = set()
            unique_tracks = []
            for track in queue_data.get("queue", []):
                if track["id"] not in seen:
                    seen.add(track["id"])
                    unique_tracks.append(track)
                    print(track["name"])

            return {"queue": unique_tracks}

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Spotify API error: {e.response.text}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

CACHE_DIR = "/tmp/smart_radio_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_filepath(query: str, track_id: str = None) -> str:
    if track_id and track_id.strip():
        clean_id = re.sub(r'[^\w]', '', track_id.strip())
        return os.path.join(CACHE_DIR, f"track_{clean_id}.mp3")
    import hashlib
    norm = normalize_key(query)
    h = hashlib.md5(norm.encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"track_{h}.mp3")

@app.post("/cache/check")
async def check_cache_status(queries: list[str] = Body(...)):
    results = {}
    for query in queries:
        filepath = get_cache_filepath(query)
        is_cached = os.path.exists(filepath) and os.path.getsize(filepath) > 50000
        results[query] = is_cached
    return {"cached": results}

@app.get("/proxy/download")
async def proxy_download_track(track: str = Query(...), trackId: str = Query(None)):
    """Download a track via local yt-dlp and return the MP3 file."""
    import static_ffmpeg
    from fastapi.responses import Response
    
    ffmpeg_bin, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    
    filepath = get_cache_filepath(track, trackId)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 50000:
        print(f"[PROXY] Cache hit for '{track}' (ID: {trackId}) -> {filepath}")
        with open(filepath, "rb") as f:
            return Response(content=f.read(), media_type="audio/mpeg")
    
    print(f"[PROXY] Downloading '{track}' for remote Modal server...")
    await pre_download_track(track, ffmpeg_bin, trackId)
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 50000:
        print(f"[PROXY] Success, serving {os.path.getsize(filepath) / 1024:.1f} KB for '{track}'")
        with open(filepath, "rb") as f:
            return Response(content=f.read(), media_type="audio/mpeg")
    
    raise HTTPException(status_code=404, detail=f"Could not download track: {track}")

def _resolve_yt_url_sync(query: str) -> str:
    import yt_dlp
    ydl_opts = {'format': 'bestaudio/best', 'quiet': True, 'no_warnings': True, 'default_search': 'ytsearch1'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=False)
        return info['entries'][0]['url'] if 'entries' in info else info['url']

async def pre_download_track(query: str, ffmpeg_bin: str, track_id: str = None) -> str:
    if not query or not query.strip():
        return None
    
    filepath = get_cache_filepath(query, track_id)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 50000:
        print(f"[LOCAL CACHE HIT] Track '{query}' (ID: {track_id}) is cached -> {filepath}")
        return filepath

    print(f"[LOCAL PRE-DOWNLOAD] Downloading track to disk cache: '{query}' (ID: {track_id})...")
    try:
        resolved_url = await asyncio.to_thread(_resolve_yt_url_sync, query)
        
        cmd = [
            ffmpeg_bin,
            '-y',
            '-reconnect', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '5',
            '-user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            '-i', resolved_url,
            '-vn',
            '-acodec', 'libmp3lame',
            '-b:a', '128k',
            '-ar', '44100',
            '-ac', '2',
            filepath
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await proc.wait()
        if os.path.exists(filepath) and os.path.getsize(filepath) > 50000:
            print(f"[LOCAL PRE-DOWNLOAD SUCCESS] Cached '{query}' ({os.path.getsize(filepath) / 1024:.1f} KB) -> {filepath}")
            return filepath
    except Exception as e:
        import traceback
        print(f"[LOCAL PRE-DOWNLOAD ERROR] Failed to cache '{query}': {e}\n{traceback.format_exc()}")
    return None

    print(f"[LOCAL PRE-DOWNLOAD] Downloading track to disk cache: '{query}'...")
    try:
        resolved_url = await asyncio.to_thread(_resolve_yt_url_sync, query)
        
        cmd = [
            ffmpeg_bin,
            '-y',
            '-reconnect', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '5',
            '-user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            '-i', resolved_url,
            '-vn',
            '-acodec', 'libmp3lame',
            '-b:a', '128k',
            '-ar', '44100',
            '-ac', '2',
            filepath
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await proc.wait()
        if os.path.exists(filepath) and os.path.getsize(filepath) > 50000:
            print(f"[LOCAL PRE-DOWNLOAD SUCCESS] Cached '{query}' ({os.path.getsize(filepath) / 1024:.1f} KB) -> {filepath}")
            return filepath
    except Exception as e:
        import traceback
        print(f"[LOCAL PRE-DOWNLOAD ERROR] Failed to cache '{query}': {e}\n{traceback.format_exc()}")
    return None

async def stream_live_url(query: str, ffmpeg_bin: str, request: Request):
    try:
        resolved_url = await asyncio.to_thread(_resolve_yt_url_sync, query)

        cmd = [
            ffmpeg_bin,
            '-reconnect', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '5',
            '-user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            '-i', resolved_url,
            '-vn',
            '-acodec', 'libmp3lame',
            '-b:a', '128k',
            '-ar', '44100',
            '-ac', '2',
            '-f', 'mp3',
            'pipe:1'
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        while True:
            if await request.is_disconnected():
                proc.kill()
                break
            chunk = await proc.stdout.read(8192)
            if not chunk:
                if proc.returncode is not None:
                    break
                await asyncio.sleep(0.05)
                continue
            yield chunk
        await proc.wait()
    except Exception as e:
        print(f"[STREAM LIVE FALLBACK ERROR] {e}")

@app.post("/tts/schedule")
async def schedule_tts_for_track(trackKey: str = Body(...), hostText: str = Body(...), trackId: str = Body(None)):
    """Schedule host speech to be injected before a specific track in the stream."""
    if not hostText or not hostText.strip():
        return {"status": "skipped"}

    text_clean = hostText.strip()
    # Save under both normalized trackKey and trackId if provided
    if trackKey:
        norm_title_key = normalize_key(trackKey)
        pending_speech[norm_title_key] = text_clean
        print(f"[TTS SCHEDULE] Scheduled speech for title key '{norm_title_key}': '{text_clean[:40]}...'")
    
    if trackId:
        norm_id_key = normalize_key(trackId)
        pending_speech[norm_id_key] = text_clean
        print(f"[TTS SCHEDULE] Scheduled speech for ID key '{norm_id_key}': '{text_clean[:40]}...'")

    # Kick off TTS pre-synthesis
    cache_path = get_tts_cache_path(text_clean)
    if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000):
        if cache_path not in _tts_in_progress:
            _tts_in_progress[cache_path] = True
            task = GPUTask(text_clean)
            await app.gpu_queue.put(task)

    return {"status": "scheduled", "trackKey": trackKey, "trackId": trackId}

@app.post("/tts/prewarm")
async def prewarm_tts(hostText: str = Body(...)):
    """Pre-synthesize TTS speech into the text-based cache so it's instant when the stream needs it."""
    if not hostText or not hostText.strip():
        return {"status": "skipped", "reason": "empty text"}
    
    cache_path = get_tts_cache_path(hostText.strip())
    
    # Already cached — instant response
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"[TTS PREWARM] Already cached: '{hostText[:50]}...'")
        return {"status": "cached", "path": cache_path}
    
    # Deduplicate: if already being synthesized, skip
    if cache_path in _tts_in_progress:
        print(f"[TTS PREWARM] Already in progress: '{hostText[:50]}...'")
        return {"status": "in_progress"}
    
    print(f"[TTS PREWARM] Queuing synthesis for: '{hostText[:60]}...'")
    _tts_in_progress[cache_path] = True
    task = GPUTask(hostText.strip())
    await app.gpu_queue.put(task)
    # Don't wait — fire and forget so the frontend isn't blocked
    return {"status": "queued", "task_id": task.task_id}

@app.get("/stream/duration")
async def get_stream_duration(track: str = Query(...), hostText: str = Query(None)):
    from pydub import AudioSegment
    speech_duration_sec = 0.0
    if hostText and hostText.strip():
        cache_path = get_tts_cache_path(hostText.strip())
        # Use cached TTS if available, otherwise synthesize
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
            print(f"[TTS CACHE HIT] Using cached TTS for duration calc")
            try:
                seg = AudioSegment.from_file(cache_path)
                speech_duration_sec = len(seg) / 1000.0
            except Exception:
                speech_duration_sec = 8.0
        else:
            speech_text = hostText.strip()
            task = GPUTask(speech_text)
            await app.gpu_queue.put(task)
            try:
                await asyncio.wait_for(task.completion_event.wait(), timeout=15.0)
                if os.path.exists(task.result_file):
                    seg = AudioSegment.from_file(task.result_file)
                    speech_duration_sec = len(seg) / 1000.0
            except Exception:
                speech_duration_sec = 8.0
            
    cached_file = get_cache_filepath(track)
    song_duration_sec = 0.0
    if os.path.exists(cached_file) and os.path.getsize(cached_file) > 50000:
        try:
            seg = AudioSegment.from_file(cached_file)
            song_duration_sec = len(seg) / 1000.0
        except Exception:
            song_duration_sec = 0.0

    return {
        "track": track,
        "speech_duration_sec": speech_duration_sec,
        "song_duration_sec": song_duration_sec,
        "total_segment_sec": speech_duration_sec + song_duration_sec
    }

@app.get("/cache/status")
async def check_cache_status(tracks: str = Query(""), trackIds: str = Query("")):
    """Returns a list of tracks that are 100% cached on disk."""
    if not tracks:
        return {"cached_tracks": []}
    
    delim = "|||" if "|||" in tracks else ","
    track_list = [t.strip() for t in tracks.split(delim) if t.strip()]
    
    track_ids_list = []
    if trackIds and trackIds.strip():
        delim_id = "|||" if "|||" in trackIds else ","
        track_ids_list = [i.strip() for i in trackIds.split(delim_id) if i.strip()]

    cached_tracks = []
    
    for idx, trk in enumerate(track_list):
        trk_id = track_ids_list[idx] if idx < len(track_ids_list) else None
        cached_file_id = get_cache_filepath(trk, trk_id) if trk_id else None
        cached_file_title = get_cache_filepath(trk, None)

        is_cached = False
        if cached_file_id and os.path.exists(cached_file_id) and os.path.getsize(cached_file_id) > 100000:
            is_cached = True
        elif os.path.exists(cached_file_title) and os.path.getsize(cached_file_title) > 100000:
            is_cached = True

        if is_cached:
            cached_tracks.append(trk)
            
    return {"cached_tracks": cached_tracks}

@app.get("/stream/live.mp3")
async def stream_live(request: Request, tracks: str = Query(None), trackIds: str = Query(None), track: str = Query(None), nextTrack: str = Query(None), thirdTrack: str = Query(None), hostText: str = Query(None)):
    from fastapi.responses import StreamingResponse
    import static_ffmpeg
    
    # Support ||| delimiter to avoid splitting artist names with commas (e.g., "Crosby, Stills & Nash")
    track_list = []
    track_ids_list = []
    if tracks and tracks.strip():
        delim = "|||" if "|||" in tracks else ","
        track_list = [t.strip() for t in tracks.split(delim) if t.strip()]
    elif track and track.strip():
        track_list.append(track.strip())
        if nextTrack and nextTrack.strip():
            track_list.append(nextTrack.strip())
        if thirdTrack and thirdTrack.strip():
            track_list.append(thirdTrack.strip())

    if trackIds and trackIds.strip():
        delim_id = "|||" if "|||" in trackIds else ","
        track_ids_list = [i.strip() for i in trackIds.split(delim_id) if i.strip()]

    print(f"\n[LOCAL API STREAM] 📻 Continuous Broadcast Requested for {len(track_list)} tracks (IDs: {len(track_ids_list)})")
    ffmpeg_bin, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()

    async def generate_chunks():
        # Stream Audio File Helper Function (Direct Disk Reader at 128kbps broadcast rate)
        async def stream_file_path(path: str):
            if not os.path.exists(path):
                return
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    yield chunk
                    await asyncio.sleep(0.05)

        # Initial host speech intro for Track 1 if passed directly in hostText param
        if hostText and hostText.strip():
            cache_path = get_tts_cache_path(hostText.strip())
            if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000):
                task = GPUTask(hostText.strip())
                await app.gpu_queue.put(task)
                try:
                    await asyncio.wait_for(task.completion_event.wait(), timeout=15.0)
                except Exception:
                    pass

            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
                print(f"[STREAM] 🎙️ Streaming intro speech: '{hostText[:40]}...'")
                async for chunk in stream_file_path(cache_path):
                    yield chunk

        # Stream all tracks continuously in a single HTTP response
        for idx, trk in enumerate(track_list):
            if await request.is_disconnected():
                print(f"[STREAM] Client disconnected at track #{idx+1} ({trk})")
                break

            # 1. Background pre-download upcoming tracks to disk by Track ID
            trk_id = track_ids_list[idx] if idx < len(track_ids_list) else None
            asyncio.create_task(pre_download_track(trk, ffmpeg_bin, trk_id))
            if idx + 1 < len(track_list):
                next_id = track_ids_list[idx + 1] if idx + 1 < len(track_ids_list) else None
                asyncio.create_task(pre_download_track(track_list[idx + 1], ffmpeg_bin, next_id))
            if idx + 2 < len(track_list):
                third_id = track_ids_list[idx + 2] if idx + 2 < len(track_ids_list) else None
                asyncio.create_task(pre_download_track(track_list[idx + 2], ffmpeg_bin, third_id))

            # 2. Check if host speech was scheduled for this track (check ID key first, then title key)
            trk_id_key = normalize_key(trk_id) if trk_id else ""
            trk_title_key = normalize_key(trk)

            scheduled_text = (pending_speech.get(trk_id_key) if trk_id_key else None) or pending_speech.get(trk_title_key)

            if scheduled_text:
                speech_cache = get_tts_cache_path(scheduled_text)
                for _ in range(16):
                    if os.path.exists(speech_cache) and os.path.getsize(speech_cache) > 1000:
                        break
                    await asyncio.sleep(0.5)

                if os.path.exists(speech_cache) and os.path.getsize(speech_cache) > 1000:
                    print(f"[STREAM] 🎙️ Injecting scheduled speech before Track #{idx+1}: '{trk}' (ID: {trk_id})")
                    async for chunk in stream_file_path(speech_cache):
                        yield chunk
                    if trk_id_key:
                        pending_speech.pop(trk_id_key, None)
                    pending_speech.pop(trk_title_key, None)
                    print(f"[STREAM] ✅ Speech injection complete for '{trk}'")
                else:
                    print(f"[STREAM] ⚠️ Speech for '{trk}' not ready in time, skipping")

            # 3. Ensure track MP3 is downloaded & cached on disk by Track ID
            cached_file = get_cache_filepath(trk, trk_id)
            if not os.path.exists(cached_file) or os.path.getsize(cached_file) < 50000:
                print(f"[STREAM] Waiting for pre-download of Track #{idx+1}: '{trk}' (ID: {trk_id})...")
                await pre_download_track(trk, ffmpeg_bin, trk_id)

            if os.path.exists(cached_file) and os.path.getsize(cached_file) > 50000:
                song_bytes = os.path.getsize(cached_file)
                print(f"[STREAM] 🎵 Streaming Track #{idx+1}/{len(track_list)}: '{trk}' ({song_bytes} bytes)")
                chunks_count = 0
                async for chunk in stream_file_path(cached_file):
                    chunks_count += 1
                    yield chunk
                print(f"[STREAM] Finished Track #{idx+1}: '{trk}' ({chunks_count} chunks)")
            else:
                print(f"[STREAM] Streaming Track #{idx+1} via live fallback pipe...")
                async for chunk in stream_live_url(trk, ffmpeg_bin, request):
                    yield chunk

    return StreamingResponse(
        generate_chunks(),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )