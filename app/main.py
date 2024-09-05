

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import FileResponse
from pydantic import BaseModel

from typing import List


from functions.classes import Track
from functions.audio_gen import get_audio
from functions.text_gen import generate_prompt, generate_text_for_song, get_llm

from TTS.api import TTS


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
    model_path="./functions/xtts_fine-tuned", 
    config_path="./functions/xtts_fine-tuned/config.json",
    gpu=True)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/radio")
def generate_radio():
    return {"Hello": "World111"}




@app.post("/get_radio_text")
def get_queue_radio(queue: List[Track]):
    print("coucou!")
    # queue = [queue[0]] # for now, just get the first track
    prompts = [generate_prompt(song_infos) for song_infos in queue]
    #songs_texts = [generate_text_for_song( prompt, llm) for prompt in prompts]
    songs_texts = generate_text_for_song( prompts, llm) 
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
        "text": songs_texts,
        "audio": "empty",
    }
    #final_queue.append(radio)

    #i=0
    #headers = {'Content-Disposition': f'attachment; filename="{f"{i}.wav"}"'}
    #return FileResponse(f"./stanley6.wav", headers=headers, media_type="audio/wav")
    #return final_queue
    return radio
@app.post("/get_radio_audio/")
def generate_audio_from_text(textForAudio: str = Body(...)):
    print("Genereting audio")
    print(textForAudio)
    get_audio(textForAudio, tts_stanley_fine_tuned)#generates the audio in output.wav


    #save_audio(response, "output.wav")
    headers = {'Content-Disposition': f'attachment; filename="{f"output.wav"}"'}
    return FileResponse(f"./output.wav", headers=headers, media_type="audio/wav")

