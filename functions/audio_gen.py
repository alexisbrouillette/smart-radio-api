from kokoro import KPipeline
import soundfile as sf
import os
import threading
import numpy as np

# Lock to ensure thread-safety during pipeline inference
pipeline_lock = threading.Lock()

# Initialize Kokoro English pipeline
pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

def get_audio(text: str, output_file: str, lang: str = 'en'):
    """
    Generates a WAV audio file from text using Kokoro-82M English model.
    """
    # Strip any potential language prefixes
    clean_text = text
    if text.startswith("fr:"):
        clean_text = text[3:].strip()
    elif text.startswith("en:"):
        clean_text = text[3:].strip()

    print(f"Synthesizing audio using Kokoro (English): '{clean_text[:60]}...'")
    with pipeline_lock:
        try:
            # Generator yields (graphemes, phonemes, audio)
            generator = pipeline(clean_text, voice='am_michael', speed=1.0)
            
            audios = []
            for _, _, audio in generator:
                if audio is not None and len(audio) > 0:
                    audios.append(audio)
            
            if not audios:
                raise ValueError("Kokoro did not generate any audio arrays.")
            
            # Concatenate audio chunks
            full_audio = np.concatenate(audios)
            
            # Save the WAV file (Kokoro's native sample rate is 24000Hz)
            sf.write(output_file, full_audio, 24000)
            print(f"Successfully saved Kokoro audio to {output_file}")
            return output_file
        except Exception as e:
            print(f"Error during Kokoro audio generation: {e}")
            raise e