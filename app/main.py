import asyncio
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
import os
import torch
import gc
from concurrent.futures import ThreadPoolExecutor
import uuid

# Set before importing TTS
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous CUDA ops
torch.backends.cudnn.benchmark = False    # Deterministic behavior
torch.backends.cudnn.deterministic = True

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow requests from this origin
    allow_credentials=True,
    allow_methods=["*"],  # allow all methods
    allow_headers=["*"],  # allow all headers
)

llm = get_llm()

tts_stanley_fine_tuned = TTS(
    model_path="./functions/xtts_fine_tuned_2", 
    config_path="./functions/xtts_fine_tuned_2/config.json",
    gpu=True)

# Create a queue for GPU tasks and a single-threaded executor
app.gpu_queue = asyncio.Queue()
app.gpu_executor = ThreadPoolExecutor(max_workers=1)  # Single thread for GPU operations

class GPUTask:
    def __init__(self, text_for_audio: str):
        self.text_for_audio = text_for_audio
        self.task_id = str(uuid.uuid4())
        self.result_file = f"output_{self.task_id}.wav"
        self.completion_event = asyncio.Event()
        self.error = None

async def gpu_worker():
    """Worker that processes GPU tasks one at a time"""
    print("Starting GPU Worker")
    while True:
        task = await app.gpu_queue.get()
        print(f"Processing GPU task: {task.task_id} (queue size: {app.gpu_queue.qsize()})")
        
        loop = asyncio.get_event_loop()
        try:
            # Execute the synchronous GPU function in our single-threaded executor
            await loop.run_in_executor(
                app.gpu_executor, 
                execute_gpu_task, 
                task.text_for_audio, 
                task.result_file
            )
            print(f"GPU task {task.task_id} completed successfully")
        except Exception as e:
            print(f"GPU task {task.task_id} failed: {e}")
            task.error = e
        finally:
            task.completion_event.set()  # Signal completion
            app.gpu_queue.task_done()

def execute_gpu_task(text_for_audio: str, output_file: str):
    """Synchronous function that runs the actual GPU operation"""
    try:
        print(f"Executing GPU task for text: {text_for_audio[:50]}...")
        
        # Call your synchronous get_audio function
        get_audio(text_for_audio, tts_stanley_fine_tuned, output_file)
        
        # Clean up GPU memory after each task
        gc.collect()
        torch.cuda.empty_cache()
        
        return output_file
    except Exception as e:
        print(f"Error in GPU task execution: {e}")
        raise

@app.on_event("startup")
async def start_workers():
    asyncio.create_task(gpu_worker())

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/radio")
def generate_radio():
    return {"Hello": "World111"}

@app.post("/get_radio_text")
def get_queue_radio(queue: List[Track]):
    print("coucou!")
    print("Queue: ", queue)
    songs_texts = generate_text_for_song(queue, llm) 
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
    """Generate audio using GPU queue with proper async coordination"""
    try:
        # Create a task with completion event
        task = GPUTask(textForAudio)
        
        # Add to queue
        await app.gpu_queue.put(task)
        print(f"Queued GPU task: {task.task_id}")
        
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
            
            # Optional: Clean up file after response (you might want to keep this for debugging)
            # @app.on_event("shutdown")
            # async def cleanup():
            #     if os.path.exists(task.result_file):
            #         os.remove(task.result_file)
            
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