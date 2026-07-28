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

class GPUTask:
    def __init__(self, text_for_audio: str):
        self.text_for_audio = text_for_audio
        self.task_id = str(uuid.uuid4())
        self.result_file = f"output_{self.task_id}.mp3"
        self.completion_event = asyncio.Event()
        self.error = None

async def gpu_worker():
    """Worker that processes TTS tasks one at a time"""
    print("Starting TTS Worker")
    while True:
        task = await app.gpu_queue.get()
        print(f"Processing TTS task: {task.task_id} (queue size: {app.gpu_queue.qsize()})")
        
        loop = asyncio.get_event_loop()
        try:
            # Execute the synchronous function in our executor
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
            task.completion_event.set()  # Signal completion
            app.gpu_queue.task_done()

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

def get_cache_filepath(query: str) -> str:
    import hashlib
    h = hashlib.md5(query.strip().lower().encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.mp3")

@app.post("/cache/check")
async def check_cache_status(queries: list[str] = Body(...)):
    results = {}
    for query in queries:
        filepath = get_cache_filepath(query)
        is_cached = os.path.exists(filepath) and os.path.getsize(filepath) > 50000
        results[query] = is_cached
    return {"cached": results}

async def pre_download_track(query: str, ffmpeg_bin: str) -> str:
    if not query or not query.strip():
        return None
    
    filepath = get_cache_filepath(query)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 50000:
        print(f"[LOCAL CACHE HIT] Track '{query}' is already cached on disk -> {filepath}")
        return filepath

    import yt_dlp
    print(f"[LOCAL PRE-DOWNLOAD] Downloading track to disk cache: '{query}'...")
    try:
        ydl_opts = {'format': 'bestaudio/best', 'quiet': True, 'no_warnings': True, 'default_search': 'ytsearch1'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            resolved_url = info['entries'][0]['url'] if 'entries' in info else info['url']
        
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
        print(f"[LOCAL PRE-DOWNLOAD ERROR] Failed to cache '{query}': {e}")
    return None

async def stream_live_url(query: str, ffmpeg_bin: str, request: Request):
    import yt_dlp
    try:
        ydl_opts = {'format': 'bestaudio/best', 'quiet': True, 'no_warnings': True, 'default_search': 'ytsearch1'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            resolved_url = info['entries'][0]['url'] if 'entries' in info else info['url']

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

@app.get("/stream/duration")
async def get_stream_duration(track: str = Query(...), hostText: str = Query(None)):
    from pydub import AudioSegment
    speech_duration_sec = 0.0
    if hostText and hostText.strip():
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
async def check_cache_status(tracks: str = Query("")):
    """Returns a list of tracks that are 100% cached on disk."""
    if not tracks:
        return {"cached_tracks": []}
    
    track_list = [t.strip() for t in tracks.split(",") if t.strip()]
    cached_tracks = []
    
    for trk in track_list:
        cached_file = get_cache_filepath(trk)
        if os.path.exists(cached_file) and os.path.getsize(cached_file) > 100000:
            cached_tracks.append(trk)
            
    return {"cached_tracks": cached_tracks}

@app.get("/stream/live.mp3")
async def stream_live(request: Request, track: str = Query(...), nextTrack: str = Query(None), thirdTrack: str = Query(None), hostText: str = Query(None)):
    from fastapi.responses import StreamingResponse
    import static_ffmpeg
    
    print(f"\n[LOCAL API STREAM] 📻 Incoming Request -> Track: '{track}' | Next: '{nextTrack}' | HostText: '{hostText[:40] if hostText else None}'")
    ffmpeg_bin, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()

    async def generate_chunks():
        tts_task_ref = {"task": None}

        async def start_bg_tasks():
            # 1. Background pre-download upcoming 3 tracks to disk in parallel!
            asyncio.create_task(pre_download_track(track, ffmpeg_bin))
            if nextTrack and nextTrack.strip():
                asyncio.create_task(pre_download_track(nextTrack, ffmpeg_bin))
            if thirdTrack and thirdTrack.strip():
                asyncio.create_task(pre_download_track(thirdTrack, ffmpeg_bin))

            # 2. Background synthesize DJ host speech
            speech_text = hostText
            if speech_text and speech_text.strip():
                try:
                    print(f"[LOCAL API STREAM] Synthesizing background TTS: '{speech_text[:60]}...'")
                    task = GPUTask(speech_text)
                    tts_task_ref["task"] = task
                    await app.gpu_queue.put(task)
                except Exception as e:
                    print(f"[LOCAL API STREAM] Background TTS error: {e}")

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

        # 1. Start background pre-downloads & TTS synthesis FIRST
        await start_bg_tasks()

        # 2. Ensure Song A is pre-downloaded on disk before streaming starts!
        cached_song_a = get_cache_filepath(track)
        if not os.path.exists(cached_song_a) or os.path.getsize(cached_song_a) < 50000:
            print(f"[LOCAL API STREAM] Waiting for Song A pre-download: '{track}'...")
            await pre_download_track(track, ffmpeg_bin)

        # 3. Stream AI Host Speech Intro FIRST (If speech task exists)
        task = tts_task_ref["task"]
        if task:
            try:
                print(f"[DEBUG LOG] Waiting for Kokoro TTS task completion...")
                await asyncio.wait_for(task.completion_event.wait(), timeout=20.0)
                if os.path.exists(task.result_file):
                    speech_bytes = os.path.getsize(task.result_file)
                    print(f"[DEBUG LOG] === BEGIN STREAMING DJ SPEECH ({speech_bytes} bytes) ===")
                    chunks_count = 0
                    async for chunk in stream_file_path(task.result_file):
                        chunks_count += 1
                        yield chunk
                    print(f"[DEBUG LOG] === FINISHED STREAMING DJ SPEECH ({chunks_count} chunks) ===")
            except Exception as e:
                print(f"[DEBUG LOG] Host speech error: {e}")

        # 4. Stream Song A directly from 0ms disk cache!
        if os.path.exists(cached_song_a) and os.path.getsize(cached_song_a) > 50000:
            song_bytes = os.path.getsize(cached_song_a)
            print(f"[DEBUG LOG] === BEGIN STREAMING SONG A: '{track}' ({song_bytes} bytes) ===")
            chunks_count = 0
            async for chunk in stream_file_path(cached_song_a):
                chunks_count += 1
                yield chunk
            print(f"[DEBUG LOG] === FINISHED STREAMING SONG A ({chunks_count} chunks) ===")
        else:
            print(f"[DEBUG LOG] === BEGIN STREAMING SONG A VIA LIVE PIPE FALLBACK ===")
            async for chunk in stream_live_url(track, ffmpeg_bin, request):
                yield chunk
            print(f"[DEBUG LOG] === FINISHED STREAMING SONG A VIA LIVE PIPE ===")

        # 3. Stream Song B (Pre-downloaded on disk)
        if nextTrack and nextTrack.strip():
            cached_song_b = get_cache_filepath(nextTrack)
            if not os.path.exists(cached_song_b):
                print(f"[LOCAL API STREAM] Song B not fully cached yet, fetching -> '{nextTrack}'")
                await pre_download_track(nextTrack, ffmpeg_bin)

            if os.path.exists(cached_song_b):
                print(f"[LOCAL API STREAM] Streaming Song B directly from 0ms disk cache -> '{nextTrack}'")
                async for chunk in stream_file_path(cached_song_b):
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