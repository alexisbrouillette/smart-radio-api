import gc
import torch
import threading

# Global lock - add this at module level
tts_lock = threading.Lock()

def get_audio(text, tts, output_file=None):
    with tts_lock:  # Only one TTS operation at a time
        try:
            result = tts.tts_to_file(
                text=text,
                speaker_wav="./functions/xtts_fine-tuned/stanley4.wav",
                language="fr",
                file_path=output_file
            )
            #del tts_stanley_fine_tuned
            return result
        except Exception as e:
            torch.cuda.empty_cache()  # Clear CUDA cache on error
            raise e

    


def save_audio(response, file_name):
    # Writing the audio stream to the file

    with open(file_name, "w") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"A new audio file was saved successfully at {file_name}")