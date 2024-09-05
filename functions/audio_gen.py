import gc
import torch

def get_audio(text, tts):


    #text = text[:10]
    tts.tts_to_file(text=text,
                speaker_wav="./functions/xtts_fine-tuned/stanley4.wav",
                language="fr")
    
    
    # del tts_stanley_fine_tuned
    # gc.collect()
    # torch.cuda.empty_cache()

def save_audio(response, file_name):
    # Writing the audio stream to the file

    with open(file_name, "w") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"A new audio file was saved successfully at {file_name}")