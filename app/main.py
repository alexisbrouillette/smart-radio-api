

from fastapi import FastAPI, Body, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import FileResponse
import httpx
from pydantic import BaseModel

from typing import List

from TTS.api import TTS
from functions.classes import Track
from functions.audio_gen import get_audio
from functions.text_gen import generate_prompt, generate_text_for_song, get_llm
import threading
import queue
from functools import wraps

import ssl

app = FastAPI()

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('app/cert.pem', keyfile='app/key.pem')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow requests from this origin
    allow_credentials=True,
    allow_methods=["*"],  # allow all methods
    allow_headers=["*"],  # allow all headers
)

llm = get_llm()

tts_stanley_fine_tuned = TTS(
    model_path="./functions/xtts_fine-tuned", 
    config_path="./functions/xtts_fine-tuned/config.json",
    gpu=True)


# Create a queue and a lock for thread-safe operations
request_queue = queue.Queue()
processing_lock = threading.Lock()

def process_queue():
    """Process items from the queue one at a time."""
    while True:
        # Get the next request (this blocks until one is available)
        item = request_queue.get()
        func, args, kwargs, future = item
        
        try:
            # Process the request
            with processing_lock:
                result = func(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            # Mark the task as done
            request_queue.task_done()

# Start the queue processing thread
queue_thread = threading.Thread(target=process_queue, daemon=True)
queue_thread.start()

def queue_request(func):
    """Decorator to add API functions to the processing queue."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        # Create a future to hold the result
        future = FutureResult()
        
        # Put the request in the queue
        request_queue.put((func, args, kwargs, future))
        
        # Return the future's result (will block until processed)
        return future.get_result()
    return wrapped

class FutureResult:
    """Simple future implementation to get async results."""
    def __init__(self):
        self._result = None
        self._exception = None
        self._event = threading.Event()

    def set_result(self, result):
        self._result = result
        self._event.set()

    def set_exception(self, exception):
        self._exception = exception
        self._event.set()

    def get_result(self):
        self._event.wait()
        if self._exception:
            raise self._exception
        return self._result


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/radio")
def generate_radio():
    return {"Hello": "World111"}



@queue_request
@app.post("/get_radio_text")
def get_queue_radio(queue: List[Track]):
    print("coucou!")
    # queue = [queue[0]] # for now, just get the first track
    print("Queue: ", queue)
    #prompts = [generate_prompt(song_infos) for song_infos in queue]
    #songs_texts = [generate_text_for_song( prompt, llm) for prompt in prompts]
    songs_texts = generate_text_for_song(queue, llm) 
    print("Generated text: ", songs_texts)
    # audios = []
    # for i, song_text in enumerate(songs_texts):
    #     response = get_audio(song_text)
    #     audios.append(response)
    #     save_audio(response, f"{i}.mp3")
    # print("Saved audio files: ")
    # print(audios)

    #final_queue = []
    #for i, track in enumerate(queue):
        # if i%3 == 0:
    lastTrack = queue[-1]
    radio = {
        "beforeTrackId": lastTrack.id,
        "afterTrackId": queue[0].id,
        "text": songs_texts,
        "audio": "empty",
    }
    #final_queue.append(radio)

    return radio

@queue_request
@app.post("/get_radio_audio")
def generate_audio_from_text(textForAudio: str = Body(...)):
    try:
        print("Genereting audio")
        print(textForAudio)
        get_audio(textForAudio, tts_stanley_fine_tuned)#generates the audio in output.wav


        #save_audio(response, "output.wav")
        headers = {'Content-Disposition': f'attachment; filename="{f"output.wav"}"'}
        return FileResponse(f"./output.wav", headers=headers, media_type="audio/wav")
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