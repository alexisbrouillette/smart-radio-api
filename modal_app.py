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
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "nodejs", "espeak-ng")
    .pip_install(
        # Core server
        "fastapi==0.112",
        "uvicorn==0.30",
        "httpx",
        "pydantic",
        "pydub",
        "yt-dlp[default]",
        "pyexecjs",
        # Coqui XTTS-v2 Stanley TTS
        "coqui-tts",
        "soundfile",
        "numpy>=1.24.0",
        "torch",
        "transformers",
        "scipy",
        # Gemini + web search
        "google-genai>=2.0.0",
        "googlesearch-python",
        "crawl4ai",
        # Crawler deps
        "nest-asyncio",
    )
    .add_local_file("youtube_cookies.txt", "/root/youtube_cookies.txt")
)

app = modal.App("smart-radio-api", image=image)

# ---------------------------------------------------------------------------
# Secret: stores GEMINI_API_KEY securely (set via: modal secret create smart-radio-secrets GEMINI_API_KEY=...)
# ---------------------------------------------------------------------------
secrets = [modal.Secret.from_name("smart-radio-secrets")]

# ---------------------------------------------------------------------------
# Persistent volumes: /audio for cached stream files, /model for Stanley XTTS weights
# ---------------------------------------------------------------------------
audio_volume = modal.Volume.from_name("smart-radio-audio", create_if_missing=True)
model_volume = modal.Volume.from_name("smart-radio-model", create_if_missing=True)
AUDIO_DIR = "/audio"
MODEL_DIR = "/model"

# ---------------------------------------------------------------------------
# FastAPI app — runs on Modal as an ASGI web endpoint
# ---------------------------------------------------------------------------
@app.function(
    gpu="t4",
    cpu=2,
    memory=4096,
    timeout=900,
    secrets=secrets,
    volumes={AUDIO_DIR: audio_volume, MODEL_DIR: model_volume},
    min_containers=0,
    scaledown_window=300,
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

    def parse_track(t: dict) -> Track:
        if isinstance(t, Track):
            return t
        t_id = str(t.get("id", ""))
        t_name = str(t.get("name", ""))
        
        raw_artists = t.get("artists", "")
        if isinstance(raw_artists, list):
            artist_names = []
            for a in raw_artists:
                if isinstance(a, dict):
                    artist_names.append(a.get("name", ""))
                elif isinstance(a, str):
                    artist_names.append(a)
            t_artists = ", ".join([name for name in artist_names if name])
        elif isinstance(raw_artists, dict):
            t_artists = str(raw_artists.get("name", ""))
        else:
            t_artists = str(raw_artists)
            
        raw_album = t.get("album", "")
        if isinstance(raw_album, dict):
            t_album = str(raw_album.get("name", ""))
            release_date = str(raw_album.get("release_date", ""))
            t_year = release_date.split("-")[0] if release_date else ""
        else:
            t_album = str(raw_album)
            t_year = str(t.get("release_year", ""))
            
        return Track(
            id=t_id,
            name=t_name,
            artists=t_artists or "Artiste inconnu",
            album=t_album or "Album",
            release_year=t_year or ""
        )

    # --------------- XTTS-v2 Stanley TTS ---------------
    from TTS.api import TTS
    import soundfile as sf
    from pydub import AudioSegment

    tts_lock = threading.Lock()
    _stanley_tts = None

    def get_stanley_tts():
        nonlocal _stanley_tts
        if _stanley_tts is None:
            import torch
            print("[XTTS] Loading fine-tuned Stanley XTTS-v2 model on GPU...")
            model_path = os.path.join(MODEL_DIR, "model.pth")
            config_path = os.path.join(MODEL_DIR, "config.json")
            vocab_path = os.path.join(MODEL_DIR, "vocab.json")
            
            if os.path.exists(model_path) and os.path.exists(config_path) and os.path.exists(vocab_path):
                try:
                    from TTS.tts.configs.xtts_config import XttsConfig
                    from TTS.tts.models.xtts import Xtts
                    config = XttsConfig()
                    config.load_json(config_path)
                    model = Xtts.init_from_config(config)
                    model.load_checkpoint(config, checkpoint_path=model_path, vocab_path=vocab_path, use_deepspeed=False)
                    if torch.cuda.is_available():
                        model.cuda()
                    _stanley_tts = (model, config)
                    print("[XTTS] ✅ Fine-tuned Stanley model loaded successfully on GPU!")
                except Exception as e:
                    print(f"[XTTS] Error loading custom model: {e}. Fallback to standard TTS wrapper.")
                    _stanley_tts = TTS(model_path=model_path, config_path=config_path, progress_bar=False, gpu=torch.cuda.is_available())
            else:
                print(f"[XTTS] Warning: files not found in {MODEL_DIR}. Fallback to standard XTTS-v2.")
                _stanley_tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
        return _stanley_tts

    def synthesize(text: str, output_file: str):
        tts_obj = get_stanley_tts()
        clean_text = text
        for prefix in ("fr:", "en:"):
            if text.startswith(prefix):
                clean_text = text[len(prefix):].strip()
        print(f"[Stanley TTS] Synthesizing French voice on GPU: '{clean_text[:60]}...'")
        
        speaker_wav = os.path.join(MODEL_DIR, "stanley4.wav")
        if not os.path.exists(speaker_wav):
            speaker_wav = None

        with tts_lock:
            temp_wav = output_file + ".tmp.wav"
            
            if isinstance(tts_obj, tuple):
                model, config = tts_obj
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_wav])
                out = model.inference(
                    text=clean_text,
                    language="fr",
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=0.75,
                    enable_text_splitting=False
                )
                import soundfile as sf
                sf.write(temp_wav, out["wav"], 24000)
            else:
                tts_obj.tts_to_file(
                    text=clean_text,
                    speaker_wav=speaker_wav,
                    language="fr",
                    file_path=temp_wav,
                    enable_text_splitting=False
                )
            
            # Broadcast Loudness Normalization & Gain Boost (+6.0 dB boost, normalized peak to -0.5 dBFS)
            try:
                seg = AudioSegment.from_file(temp_wav)
                normalized_seg = seg.apply_gain(6.0).normalize(headroom=0.5)
                mp3_file = output_file.replace('.wav', '.mp3') if output_file.endswith('.wav') else output_file
                normalized_seg.export(mp3_file, format="mp3", bitrate="128k")
                # Also save volume-boosted wav if needed
                normalized_seg.export(output_file, format="wav")
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
                print(f"[Stanley TTS] ✅ Saved volume-boosted Stanley speech to {output_file}")
            except Exception as e:
                print(f"[Stanley TTS Volume Boost Warning] {e}")
                if os.path.exists(temp_wav):
                    os.rename(temp_wav, output_file)

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
                "chanson sens paroles inspiration histoire composition",
                "Concentre-toi UNIQUEMENT sur le sens de la chanson, l'inspiration de l'artiste ou l'histoire derrière les paroles. Partage une seule anecdote humaine."
            )
        elif focus == "influences":
            return (
                "influences musicales qui a inspire cette chanson artiste",
                "Concentre-toi UNIQUEMENT sur les influences musicales de l'artiste pour cette chanson ou ses collaborations. Partage une seule anecdote humaine."
            )
        elif focus == "tour":
            return (
                "concerts tournees spectacle live actualites 2026",
                "Concentre-toi UNIQUEMENT sur les prochains concerts, tournées ou actualités scéniques récentes de l'artiste. Partage une seule anecdote humaine."
            )
        elif focus == "trivia":
            return (
                "anecdotes coulisses enregistrement production faits amusants",
                "Concentre-toi UNIQUEMENT sur une anecdote originale ou un fait amusant des coulisses de l'enregistrement ou de la production. Partage une seule anecdote humaine."
            )
        else:
            return (
                "album ambiance reception critique univers musical",
                "Concentre-toi UNIQUEMENT sur l'ambiance générale du morceau, son accueil critique ou l'univers de l'album. Partage une seule anecdote humaine."
            )

    def radio_host_prompt(prev: Track, nxt: Track, history_str: str, focus_instruction: str) -> str:
        return f"""
        Tu es Stanley, un animateur radio chevronné, chaleureux et naturel pour Smart Radio. Rédige une intervention fluide et captivante de 20 à 30 secondes en FRANÇAIS.
        
        Historique de la session (ce que tu as déjà dit pour éviter les répétitions):
        {history_str}
        
        Détails du morceau:
        - Morceau qui vient de se terminer: "{prev.name}" par {prev.artists}
        - Morceau suivant: "{nxt.name}" par {nxt.artists} (de l'album "{nxt.album}", sorti en {nxt.release_year})
        
        Instructions d'animation:
        1. Fais une rapide transition en mentionnant le morceau qui vient de se terminer ("{prev.name}" par {prev.artists}).
        2. Présente ensuite en détail le morceau suivant ("{nxt.name}" par {nxt.artists}).
        3. {focus_instruction}
        4. Partage UNIQUEMENT cette anecdote/élément de contexte. Ne résume pas toute la carrière de l'artiste.
        5. Reste un vrai animateur humain:
           - Évite le ton artificiel et les phrases toutes faites comme "morceau captivant", "chef-d'œuvre absolu".
           - Varie tes tournures et reste spontané.
        
        Règles de rédaction:
        - Rédige STRICTEMENT en français.
        - Maximum 100 mots (environ 20 à 25 secondes à l'oral).
        - Format parlé direct: n'écris pas d'actions comme [rit] ou d'indications scéniques.
        - N'utilise aucun symbole markdown comme * [ ] ( ).
        """

    def radio_host_prompt_with_data(prev: Track, nxt: Track, data: str, history_str: str, focus_instruction: str) -> str:
        return f"""
        Tu es Stanley, un animateur radio chevronné, chaleureux et naturel pour Smart Radio. Rédige une intervention fluide et captivante de 20 à 30 secondes en FRANÇAIS.
        
        Historique de la session (ce que tu as déjà dit pour éviter les répétitions):
        {history_str}
        
        Détails du morceau:
        - Morceau qui vient de se terminer: "{prev.name}" par {prev.artists}
        - Morceau suivant: "{nxt.name}" par {nxt.artists} (de l'album "{nxt.album}", sorti en {nxt.release_year})
        
        Données web trouvées sur la chanson:
        {data}
        
        Instructions d'animation:
        1. Fais une rapide transition en mentionnant le morceau qui vient de se terminer ("{prev.name}" par {prev.artists}).
        2. Présente ensuite en détail le morceau suivant ("{nxt.name}" par {nxt.artists}).
        3. {focus_instruction}
        4. Partage UNIQUEMENT cette anecdote/élément de contexte en utilisant les données ci-dessus.
        5. Reste un vrai animateur humain:
           - Évite le ton artificiel et les phrases toutes faites.
           - Varie tes ouvertures et fermetures d'antenne.
        
        Règles de rédaction:
        - Rédige STRICTEMENT en français.
        - Maximum 100 mots (environ 20 à 25 secondes à l'oral).
        - Format parlé direct: n'écris pas d'actions comme [rit] ou d'indications scéniques.
        - N'utilise aucun symbole markdown comme * [ ] ( ).
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

        tracks = [parse_track(t) for t in tracks_data]
        print(f"Received {len(tracks)} tracks and {len(history)} history items")
        try:
            text = await generate_text_for_song(tracks, history)
            if not text or not text.strip():
                raise ValueError("Empty text from Gemini")
        except Exception as e:
            print(f"Text gen failed: {e}. Using default.")
            prev = tracks[0] if tracks else None
            nxt = tracks[-1] if len(tracks) > 1 else None
            text = f"C'était {prev.name} par {prev.artists}. Place maintenant à {nxt.name} par {nxt.artists} !" if prev and nxt else "Restez à l'écoute sur Smart Radio !"

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

        tracks = [parse_track(t) for t in tracks_data]
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
            text = f"C'était {prev.name} par {prev.artists}. Place maintenant à {nxt.name} par {nxt.artists} !" if prev and nxt else "Restez à l'écoute sur Smart Radio !"

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

    SCHEDULED_HOST_TEXTS = {}

    @fast_app.post("/tts/schedule")
    async def schedule_tts(request: Request):
        try:
            data = await request.json()
            track_key = data.get("trackKey", "").strip()
            host_text = data.get("hostText", "").strip()
            track_id = data.get("trackId", "").strip()
            
            if host_text:
                if track_id:
                    SCHEDULED_HOST_TEXTS[track_id] = host_text
                if track_key:
                    SCHEDULED_HOST_TEXTS[track_key] = host_text
                print(f"[MODAL TTS SCHEDULE] Scheduled host speech for trackId={track_id}, key='{track_key}': '{host_text[:60]}...'")
            return {"status": "scheduled", "trackId": track_id}
        except Exception as e:
            print(f"[MODAL TTS SCHEDULE ERROR] {e}")
            return {"status": "error", "message": str(e)}

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

        # 1. Try YouTube / YouTube Music search with authenticated cookies
        try:
            print(f"[MODAL STREAM] Trying YouTube search with cookies for '{query}'...")
            ydl_opts_yt = {
                'format': 'bestaudio/best',
                'quiet': True,
                'no_warnings': True,
                'default_search': 'ytsearch1',
                'cookiefile': '/root/youtube_cookies.txt'
            }
            with yt_dlp.YoutubeDL(ydl_opts_yt) as ydl:
                info = ydl.extract_info(query, download=False)
                if 'entries' in info and len(info['entries']) > 0:
                    entry = info['entries'][0]
                    best_url = None
                    for fmt in entry.get('formats', []):
                        if 'googlevideo.com' in fmt.get('url', '') and fmt.get('ext') in ('m4a', 'webm', 'mp4', 'opus'):
                            best_url = fmt['url']
                            if fmt.get('vcodec') == 'none':
                                break
                    url = best_url or entry.get('url')
                    if url:
                        print(f"[MODAL STREAM] YouTube resolved stream URL for '{query}' -> {url[:80]}...")
                        return url
        except Exception as e:
            print(f"[MODAL STREAM] YouTube search error for '{query}': {e}")

        # 2. Fallback to SoundCloud Search
        try:
            print(f"[MODAL STREAM] Searching SoundCloud fallback for '{query}'...")
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
        except Exception as e:
            print(f"[MODAL STREAM] SoundCloud fallback error for '{query}': {e}")

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

    @fast_app.get("/stream/duration")
    async def get_stream_duration(track: str = Query(...), hostText: str = Query(None)):
        from pydub import AudioSegment
        speech_duration_sec = 0.0
        if hostText and hostText.strip():
            speech_text = hostText.strip()
            task = GPUTask(speech_text)
            await tts_queue.put(task)
            try:
                await asyncio.wait_for(task.event.wait(), timeout=15.0)
                if os.path.exists(task.result_file):
                    seg = AudioSegment.from_file(task.result_file)
                    speech_duration_sec = len(seg) / 1000.0
            except Exception:
                speech_duration_sec = 8.0
                
        cached_file = get_modal_cache_filepath(track)
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

                # 2. Background synthesize DJ host speech (Check URL param first, then SCHEDULED_HOST_TEXTS)
                speech_text = hostText
                if not speech_text or not speech_text.strip():
                    if nextTrack and nextTrack.strip() in SCHEDULED_HOST_TEXTS:
                        speech_text = SCHEDULED_HOST_TEXTS[nextTrack.strip()]
                    elif track and track.strip() in SCHEDULED_HOST_TEXTS:
                        speech_text = SCHEDULED_HOST_TEXTS[track.strip()]

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
                    print(f"[MODAL STREAM] Waiting max 30s for host speech intro...")
                    await asyncio.wait_for(task.event.wait(), timeout=30.0)
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
