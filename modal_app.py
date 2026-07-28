"""
Smart Radio API — Modal Deployment
Deploy with: modal deploy modal_app.py
"""

import modal
import os

# ---------------------------------------------------------------------------
# Image: build once, cached on Modal's servers
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "nodejs")
    .pip_install(
        # Core server
        "fastapi==0.112",
        "uvicorn==0.30",
        "httpx",
        "pydantic",
        "pydub",
        "yt-dlp",
        # TTS
        "kokoro",
        "soundfile",
        "numpy>=1.26",
        "torch",
        # Kokoro language dependencies
        "nltk==3.8.1",
        "Unidecode==1.3.8",
        "inflect",
        "anyascii",
        # Gemini + web search
        "google-genai>=2.0.0",
        "googlesearch-python",
        "crawl4ai",
        # Crawler deps
        "nest-asyncio",
    )
    .run_commands(
        # Download NLTK data needed by Kokoro
        "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
    )
)

app = modal.App("smart-radio-api", image=image)

# ---------------------------------------------------------------------------
# Secret: stores GEMINI_API_KEY securely (set via: modal secret create smart-radio-secrets GEMINI_API_KEY=...)
# ---------------------------------------------------------------------------
secrets = [modal.Secret.from_name("smart-radio-secrets")]

# ---------------------------------------------------------------------------
# Persistent volume for WAV output files (avoids regenerating on cold starts)
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("smart-radio-audio", create_if_missing=True)
AUDIO_DIR = "/audio"

# ---------------------------------------------------------------------------
# FastAPI app — runs on Modal as an ASGI web endpoint
# ---------------------------------------------------------------------------
@app.function(
    # Use CPU-only (no GPU needed for Kokoro-82M at acceptable speed)
    # Switch to gpu="t4" if you want faster TTS
    cpu=2,
    memory=4096,
    timeout=300,
    secrets=secrets,
    volumes={AUDIO_DIR: volume},
    # Keep 1 warm container to avoid cold-start delays mid-music
    min_containers=0,
    scaledown_window=300,  # 5-minute idle before scale-down
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    """Wraps the existing FastAPI app for Modal deployment."""
    import asyncio
    import gc
    import uuid
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import soundfile as sf
    from fastapi import FastAPI, Body, HTTPException, Header, Request, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    import httpx
    from pydantic import BaseModel
    from typing import List

    # --------------- Inline Track model (avoids importing local modules) ---------------
    class Track(BaseModel):
        id: str
        name: str
        artists: str
        album: str
        release_year: str

    # --------------- Kokoro TTS ---------------
    from kokoro import KPipeline
    pipeline_lock = threading.Lock()
    _pipeline = None

    def get_pipeline():
        nonlocal _pipeline
        if _pipeline is None:
            print("Loading Kokoro pipeline...")
            _pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
            print("Kokoro pipeline loaded.")
        return _pipeline

    def synthesize(text: str, output_file: str):
        pipeline = get_pipeline()
        clean_text = text
        for prefix in ("fr:", "en:"):
            if text.startswith(prefix):
                clean_text = text[len(prefix):].strip()
        print(f"Synthesizing: '{clean_text[:60]}...'")
        with pipeline_lock:
            generator = pipeline(clean_text, voice='am_michael', speed=1.0)
            audios = [audio for _, _, audio in generator if audio is not None and len(audio) > 0]
            if not audios:
                raise ValueError("Kokoro generated no audio arrays.")
            
            wav_temp = output_file if output_file.endswith('.wav') else output_file + '.wav'
            sf.write(wav_temp, np.concatenate(audios), 24000)
            
            # Resample to broadcast 44.1kHz stereo MP3
            from pydub import AudioSegment
            sound = AudioSegment.from_wav(wav_temp)
            sound = sound.set_frame_rate(44100).set_channels(2)
            mp3_file = output_file.replace('.wav', '.mp3') if output_file.endswith('.wav') else output_file
            sound.export(mp3_file, format="mp3", bitrate="128k")
            if os.path.exists(wav_temp) and wav_temp != mp3_file:
                os.remove(wav_temp)
            print(f"Saved 44.1kHz stereo MP3 audio to {mp3_file}")

    # --------------- Text gen ---------------
    from google import genai
    from google.genai import types
    from googlesearch import search as google_search

    def _get_client():
        return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    MODELS = ["models/gemini-3.5-flash-lite", "models/gemini-2.0-flash-lite", "models/gemini-2.0-flash"]

    def generate_text(prompt: str, use_grounding: bool = True) -> str:
        client = _get_client()
        last_err = None
        for model_name in MODELS:
            try:
                print(f"[Gemini] {model_name} (grounding={use_grounding})")
                config_kwargs = {}
                if use_grounding:
                    config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                text = response.text
                if text and text.strip():
                    print(f"[Gemini] ✅ Success with {model_name}")
                    return text
            except Exception as e:
                print(f"[Gemini] ❌ {model_name}: {e}")
                last_err = e
        if last_err:
            raise last_err
        return ""

    # --------------- Text gen ---------------
    from google import genai
    from google.genai import types
    from googlesearch import search as google_search
    import random

    def _get_client():
        return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    MODELS = ["models/gemini-3.5-flash-lite", "models/gemini-2.0-flash-lite", "models/gemini-2.0-flash"]

    def generate_text(prompt: str, use_grounding: bool = True) -> str:
        client = _get_client()
        last_err = None
        for model_name in MODELS:
            try:
                print(f"[Gemini] {model_name} (grounding={use_grounding})")
                config_kwargs = {}
                if use_grounding:
                    config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                text = response.text
                if text and text.strip():
                    print(f"[Gemini] ✅ Success with {model_name}")
                    return text
            except Exception as e:
                print(f"[Gemini] ❌ {model_name}: {e}")
                last_err = e
        if last_err:
            raise last_err
        return ""

    def get_focus_instruction(focus: str) -> tuple[str, str]:
        if focus == "meaning":
            return (
                "song meaning lyrics writing inspiration origin story interview",
                "Focus ONLY on the meaning of the song, what inspired the artist to write it, or the story behind the lyrics. Keep it to one simple, human insight."
            )
        elif focus == "influences":
            return (
                "musical influences who inspired this song artist background",
                "Focus ONLY on the artist's musical influences for this song, or who they were listening to or collaborating with when making it. Keep it to one simple, human insight."
            )
        elif focus == "tour":
            return (
                "tour dates concerts live show news 2026",
                "Focus ONLY on upcoming concerts, tour dates, or a notable recent live performance news for the artist. Keep it to one simple, human insight."
            )
        elif focus == "trivia":
            return (
                "trivia fun facts behind the scenes production recording process",
                "Focus ONLY on a single, interesting piece of behind-the-scenes trivia or fun facts about the recording, release, or production of the song. Keep it to one simple, human insight."
            )
        else:
            return (
                "album review critical reception vibe",
                "Focus ONLY on the general vibe of the song, its critical reception, or how the album was received. Keep it to one simple, human insight."
            )

    def radio_host_prompt(prev: Track, nxt: Track, history_str: str, focus_instruction: str) -> str:
        return f"""
        You are a veteran radio host with a warm, natural, and highly conversational voice. Write a smooth, engaging 20-30 second transition in English.
        
        Session history (what you already played/said in this session to avoid repeating yourself):
        {history_str}
        
        Transition details:
        - Just played: "{prev.name}" by {prev.artists}
        - Up next: "{nxt.name}" by {nxt.artists} (from the album "{nxt.album}", released in {nxt.release_year})
        
        Hosting Instructions:
        1. Search Google for info about the next song or artist.
        2. {focus_instruction}
        3. Share ONLY this single focus point. Do NOT summarize their whole career, list multiple facts, or dump general biography details. Keep it to one interesting, natural anecdote.
        4. Be a normal human host. Avoid AI slop:
           - NEVER use over-dramatic words like "captivating", "electrifying", "absolute masterpiece", "sure to get you moving".
           - NEVER wrap the transition in a neat summary/conclusion or tell the listener to "enjoy" or "dive in".
           - Do NOT repeat the sentence structure, greetings, or transitions from the session history. Avoid clichés like "Up next...", "That was...", "Here's...", "Let's keep the vibe going...". Be conversational, spontaneous, and vary your opening and closing lines.
        
        Rules:
        - Write in English.
        - Maximum 120 words.
        - Spoken format: do not write actions like [laughs] or music descriptions.
        - Do not use markdown styling or symbols like * [ ] ( ).
        - Speak like a real human host, not a generic AI assistant.
        """

    def radio_host_prompt_with_data(prev: Track, nxt: Track, data: str, history_str: str, focus_instruction: str) -> str:
        return f"""
        You are a veteran radio host with a warm, natural, and highly conversational voice. Write a smooth, engaging 20-30 second transition in English.
        
        Session history (what you already played/said in this session to avoid repeating yourself):
        {history_str}
        
        Transition details:
        - Just played: "{prev.name}" by {prev.artists}
        - Up next: "{nxt.name}" by {nxt.artists} (from the album "{nxt.album}", released in {nxt.release_year})
        
        Facts/Info crawled from web search about the next song:
        {data}
        
        Hosting Instructions:
        1. {focus_instruction}
        2. Share ONLY this single focus point using the crawled data above. Do NOT summarize their whole career, list multiple facts, or dump general biography details. Keep it to one interesting, natural anecdote.
        3. Be a normal human host. Avoid AI slop:
           - NEVER use over-dramatic words like "captivating", "electrifying", "absolute masterpiece", "sure to get you moving".
           - NEVER wrap the transition in a neat summary/conclusion or tell the listener to "enjoy" or "dive in".
           - Do NOT repeat the sentence structure, greetings, or transitions from the session history. Avoid clichés like "Up next...", "That was...", "Here's...", "Let's keep the vibe going...". Be conversational, spontaneous, and vary your opening and closing lines.
        
        Rules:
        - Write in English.
        - Maximum 120 words.
        - Spoken format: do not write actions like [laughs] or music descriptions.
        - Do not use markdown styling or symbols like * [ ] ( ).
        - Speak like a real human host, not a generic AI assistant.
        """

    async def _crawl(url: str) -> str:
        try:
            from crawl4ai import AsyncWebCrawler
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                return result.markdown[:3000]
        except Exception as e:
            return ""

    async def generate_text_for_song(tracks: List[Track], history: List[dict] = None) -> str:
        if len(tracks) < 2:
            return ""
        prev, nxt = tracks[0], tracks[1]

        # Randomly choose a focus topic to keep transitions diverse and natural
        focus = random.choice(["meaning", "influences", "tour", "trivia", "vibe"])
        query_suffix, focus_instruction = get_focus_instruction(focus)
        print(f"[TextGen] Selected focus: {focus}")

        # Format history string to pass to prompt
        if history:
            history_lines = []
            for h in history:
                history_lines.append(f'- Transition for "{h.get("song")}" by {h.get("artist")}: "{h.get("text")}"')
            history_str = "\n".join(history_lines)
        else:
            history_str = "No past announcements in this session yet."

        # Attempt 1: native grounding
        try:
            prompt = radio_host_prompt(prev, nxt, history_str, focus_instruction)
            text = generate_text(prompt, use_grounding=True)
            if text:
                print("[Strategy] ✅ Native grounding")
                return text
        except Exception as e:
            print(f"[Strategy] ❌ Grounding failed: {e}. Falling back to crawler...")

        # Attempt 2: python crawler
        try:
            query = f"{nxt.name} {nxt.artists} {query_suffix}"
            excluded = ["youtube","facebook","genius","bandcamp","amazon","spotify","apple","twitter","instagram"]
            urls = []
            for r in google_search(query, num_results=6, advanced=True):
                if not any(d in r.url for d in excluded):
                    urls.append(r.url)
                    if len(urls) >= 1:
                        break
            web_data = await _crawl(urls[0]) if urls else ""
            prompt_with_data = radio_host_prompt_with_data(prev, nxt, web_data, history_str, focus_instruction)
            text = generate_text(prompt_with_data, use_grounding=False)
            if text:
                print("[Strategy] ✅ Crawler fallback")
                return text
        except Exception as e:
            print(f"[Strategy] ❌ Crawler failed: {e}")

        return ""

    # --------------- FastAPI app ---------------
    fast_app = FastAPI(title="Smart Radio API")
    fast_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=3600,
    )

    tts_queue: asyncio.Queue = asyncio.Queue()
    executor = ThreadPoolExecutor(max_workers=1)

    class GPUTask:
        def __init__(self, text: str):
            self.text = text
            self.task_id = str(uuid.uuid4())
            self.result_file = f"{AUDIO_DIR}/output_{self.task_id}.wav"
            self.event = asyncio.Event()
            self.error = None

    async def tts_worker():
        print("TTS worker started")
        while True:
            task = await tts_queue.get()
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(executor, synthesize, task.text, task.result_file)
            except Exception as e:
                task.error = e
            finally:
                task.event.set()
                tts_queue.task_done()

    @fast_app.on_event("startup")
    async def startup():
        asyncio.create_task(tts_worker())

    @fast_app.get("/")
    def root():
        return {"status": "Smart Radio API running on Modal 🎙️"}

    @fast_app.post("/get_radio_text")
    async def get_radio_text(request: Request):
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        history = []
        if isinstance(data, list):
            # Old format: list of tracks
            tracks_data = data
        elif isinstance(data, dict):
            # New format: {"tracks": [...], "history": [...]}
            tracks_data = data.get("tracks", [])
            history = data.get("history", [])
        else:
            raise HTTPException(status_code=400, detail="Invalid request body")

        tracks = [Track(**t) for t in tracks_data]
        print(f"Received {len(tracks)} tracks and {len(history)} history items")
        try:
            text = await generate_text_for_song(tracks, history)
            if not text or not text.strip():
                raise ValueError("Empty text from Gemini")
        except Exception as e:
            print(f"Text gen failed: {e}. Using default.")
            prev = tracks[0] if tracks else None
            nxt = tracks[-1] if len(tracks) > 1 else None
            text = f"That was {prev.name} by {prev.artists}. Up next, {nxt.name} by {nxt.artists}!" if prev and nxt else "Stay tuned!"

        last_track = tracks[-1]
        return {
            "beforeTrackId": last_track.id,
            "afterTrackId": tracks[0].id,
            "text": text,
            "audio": "empty",
        }

    @fast_app.post("/get_radio_full")
    async def get_radio_full(request: Request):
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        history = []
        if isinstance(data, list):
            tracks_data = data
        elif isinstance(data, dict):
            tracks_data = data.get("tracks", [])
            history = data.get("history", [])
        else:
            raise HTTPException(status_code=400, detail="Invalid request body")

        tracks = [Track(**t) for t in tracks_data]
        if not tracks:
            raise HTTPException(status_code=400, detail="No tracks provided")

        try:
            text = await generate_text_for_song(tracks, history)
            if not text or not text.strip():
                raise ValueError("Empty text from Gemini")
        except Exception as e:
            print(f"Text gen failed: {e}. Using default.")
            prev = tracks[0] if tracks else None
            nxt = tracks[-1] if len(tracks) > 1 else None
            text = f"That was {prev.name} by {prev.artists}. Up next, {nxt.name} by {nxt.artists}!" if prev and nxt else "Stay tuned!"

        # Generate TTS WAV audio immediately
        task = GPUTask(text)
        await tts_queue.put(task)
        b64_audio = None
        try:
            await asyncio.wait_for(task.event.wait(), timeout=120.0)
            if os.path.exists(task.result_file):
                with open(task.result_file, "rb") as f:
                    import base64
                    b64_audio = base64.b64encode(f.read()).decode('utf-8')
        except Exception as err:
            print(f"TTS synth error in /get_radio_full: {err}")

        last_track = tracks[-1]
        return {
            "beforeTrackId": last_track.id,
            "afterTrackId": tracks[0].id,
            "text": text,
            "audio": b64_audio,
        }

    @fast_app.post("/get_radio_audio")
    async def get_radio_audio(textForAudio: str = Body(...)):
        task = GPUTask(textForAudio)
        await tts_queue.put(task)
        try:
            await asyncio.wait_for(task.event.wait(), timeout=300.0)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="TTS timed out")
        if task.error:
            raise HTTPException(status_code=500, detail=str(task.error))
        if not os.path.exists(task.result_file):
            raise HTTPException(status_code=500, detail="Audio file not found")
        return FileResponse(task.result_file, media_type="audio/wav")

    @fast_app.get("/spotify/queue")
    async def spotify_queue(authorization: str = Header(...)):
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.spotify.com/v1/me/player/queue",
                headers={"Authorization": authorization},
            )
            resp.raise_for_status()
            data = resp.json()
            seen, unique = set(), []
    CACHE_DIR = "/tmp/smart_radio_cache"
    os.makedirs(CACHE_DIR, exist_ok=True)

    def fetch_track_audio_url(query: str):
        import yt_dlp

        # 1. Try SoundCloud Search (Fast, 100% unblocked on cloud IPs!)
        try:
            print(f"[MODAL STREAM] Searching SoundCloud for '{query}'...")
            ydl_opts_sc = {
                'format': 'bestaudio/best',
                'quiet': True,
                'no_warnings': True,
                'default_search': 'scsearch1'
            }
            with yt_dlp.YoutubeDL(ydl_opts_sc) as ydl:
                info = ydl.extract_info(query, download=False)
                if 'entries' in info and len(info['entries']) > 0 and info['entries'][0].get('url'):
                    url = info['entries'][0]['url']
                    print(f"[MODAL STREAM] SoundCloud resolved URL for '{query}' -> {url[:80]}...")
                    return url
                elif info.get('url'):
                    url = info['url']
                    print(f"[MODAL STREAM] SoundCloud resolved URL for '{query}' -> {url[:80]}...")
                    return url
        except Exception as e:
            print(f"[MODAL STREAM] SoundCloud search error for '{query}': {e}")

        # 2. Try YouTube Search with android_music client fallback
        try:
            print(f"[MODAL STREAM] Trying YouTube search fallback for '{query}'...")
            ydl_opts_yt = {
                'format': 'bestaudio/best',
                'quiet': True,
                'no_warnings': True,
                'default_search': 'ytsearch1',
                'extractor_args': {'youtube': {'player_client': ['android_music', 'web']}}
            }
            with yt_dlp.YoutubeDL(ydl_opts_yt) as ydl:
                info = ydl.extract_info(query, download=False)
                if 'entries' in info and len(info['entries']) > 0 and info['entries'][0].get('url'):
                    return info['entries'][0]['url']
                elif info.get('url'):
                    return info['url']
        except Exception as e:
            print(f"[MODAL STREAM] YouTube search error for '{query}': {e}")

        return None

    def get_modal_cache_filepath(query: str) -> str:
        import hashlib
        h = hashlib.md5(query.strip().lower().encode('utf-8')).hexdigest()
        return os.path.join(CACHE_DIR, f"{h}.mp3")

    @fast_app.post("/cache/check")
    async def check_modal_cache_status(queries: list[str] = Body(...)):
        results = {}
        for query in queries:
            filepath = get_modal_cache_filepath(query)
            is_cached = os.path.exists(filepath) and os.path.getsize(filepath) > 50000
            results[query] = is_cached
        return {"cached": results}

    async def pre_download_modal_track(query: str) -> str:
        if not query or not query.strip():
            return None
        
        filepath = get_modal_cache_filepath(query)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 50000:
            print(f"[MODAL CACHE HIT] Track '{query}' is already cached on disk -> {filepath}")
            return filepath

        print(f"[MODAL PRE-DOWNLOAD] Pre-downloading track to disk cache: '{query}'...")
        audio_url = fetch_track_audio_url(query)
        if not audio_url:
            print(f"[MODAL PRE-DOWNLOAD ERROR] Could not resolve audio URL for '{query}'")
            return None

        try:
            cmd = [
                'ffmpeg',
                '-y',
                '-reconnect', '1',
                '-reconnect_streamed', '1',
                '-reconnect_delay_max', '5',
                '-user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                '-i', audio_url,
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
                print(f"[MODAL PRE-DOWNLOAD SUCCESS] Cached '{query}' ({os.path.getsize(filepath) / 1024:.1f} KB) -> {filepath}")
                return filepath
        except Exception as e:
            print(f"[MODAL PRE-DOWNLOAD ERROR] Failed to cache '{query}': {e}")
        return None
    @fast_app.get("/cache/status")
    async def check_cache_status(tracks: str = Query("")):
        if not tracks:
            return {"cached_tracks": []}
        
        track_list = [t.strip() for t in tracks.split(",") if t.strip()]
        cached_tracks = []
        
        for trk in track_list:
            cached_file = get_modal_cache_filepath(trk)
            if os.path.exists(cached_file) and os.path.getsize(cached_file) > 100000:
                cached_tracks.append(trk)
                
        return {"cached_tracks": cached_tracks}

    @fast_app.get("/stream/live.mp3")
    async def stream_live_radio(request: Request, track: str = "Cheikh Lo Sante Yalla", nextTrack: str = None, thirdTrack: str = None, hostText: str = None):
        from fastapi.responses import StreamingResponse

        async def generate_radio_chunks():
            tts_task_ref = {"task": None}

            async def start_bg_tasks():
                # 1. Background pre-download upcoming 3 tracks in parallel!
                asyncio.create_task(pre_download_modal_track(track))
                if nextTrack and nextTrack.strip():
                    asyncio.create_task(pre_download_modal_track(nextTrack))
                if thirdTrack and thirdTrack.strip():
                    asyncio.create_task(pre_download_modal_track(thirdTrack))

                # 2. Background synthesize DJ host speech
                speech_text = hostText

                if speech_text and speech_text.strip():
                    try:
                        print(f"[MODAL STREAM] Synthesizing background TTS for: '{speech_text[:60]}...'")
                        task = GPUTask(speech_text)
                        tts_task_ref["task"] = task
                        await tts_queue.put(task)
                    except Exception as e:
                        print(f"Background TTS trigger error: {e}")

            # Start background pre-downloads immediately!
            await start_bg_tasks()

            # Stream Audio File Helper Function (Direct Disk Reader at 128kbps broadcast rate)
            async def stream_file_path(path: str):
                if not os.path.exists(path):
                    return
                with open(path, "rb") as f:
                    while True:
                        if await request.is_disconnected():
                            break
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        yield chunk
                        await asyncio.sleep(0.05)

            # Live URL Stream Fallback Helper
            async def stream_live_url(query: str):
                audio_url = fetch_track_audio_url(query)
                if not audio_url:
                    print(f"[MODAL STREAM ERROR] Live URL fallback resolution failed for '{query}'")
                    return
                cmd = [
                    'ffmpeg',
                    '-reconnect', '1',
                    '-reconnect_streamed', '1',
                    '-reconnect_delay_max', '5',
                    '-user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    '-i', audio_url,
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

            # 1. Stream AI Host Speech Intro FIRST (If speech text is provided)
            task = tts_task_ref["task"]
            if task:
                try:
                    print(f"[MODAL STREAM] Waiting max 2.0s for host speech intro...")
                    await asyncio.wait_for(task.event.wait(), timeout=2.0)
                    if os.path.exists(task.result_file):
                        print(f"[MODAL STREAM] Streaming host speech intro...")
                        async for chunk in stream_file_path(task.result_file):
                            yield chunk
                except Exception as e:
                    print(f"Host speech intro skipped: {e}")

            # 2. Stream Song A (Check disk cache first, else fallback to live URL)
            cached_song_a = get_modal_cache_filepath(track)
            if not os.path.exists(cached_song_a):
                await pre_download_modal_track(track)

            if os.path.exists(cached_song_a):
                print(f"[MODAL STREAM] Streaming Song A from disk cache -> '{track}'")
                async for chunk in stream_file_path(cached_song_a):
                    yield chunk
            else:
                print(f"[MODAL STREAM] Disk cache miss, streaming Song A via live URL fallback -> '{track}'")
                yielded_any = False
                async for chunk in stream_live_url(track):
                    yielded_any = True
                    yield chunk
                
                if not yielded_any:
                    print(f"[MODAL STREAM FALLBACK] Yielding 2s silent MP3 frame for '{track}'...")
                    from pydub import AudioSegment
                    import io
                    silence = AudioSegment.silent(duration=2000)
                    buf = io.BytesIO()
                    silence.export(buf, format="mp3")
                    yield buf.getvalue()

            # 3. Stream Song B (Already pre-downloaded on disk!)
            if nextTrack and nextTrack.strip():
                cached_song_b = get_modal_cache_filepath(nextTrack)
                if not os.path.exists(cached_song_b):
                    print(f"[MODAL STREAM] Song B not fully cached yet, pre-downloading -> '{nextTrack}'")
                    await pre_download_modal_track(nextTrack)

                if os.path.exists(cached_song_b):
                    print(f"[MODAL STREAM] Streaming Song B directly from 0ms disk cache -> '{nextTrack}'")
                    async for chunk in stream_file_path(cached_song_b):
                        yield chunk

        return StreamingResponse(
            generate_radio_chunks(),
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Connection": "keep-alive"
            }
        )

    return fast_app
