import os
import torch
import threading
import soundfile as sf

# Fix compatibility between modern transformers and Coqui TTS
import warnings
warnings.filterwarnings('ignore')

import logging
try:
    from transformers import logging as tf_logging
    tf_logging.set_verbosity_error()
except Exception:
    pass

def _soundfile_load(filepath, **kwargs):
    data, sr = sf.read(filepath, dtype="float32")
    if data.ndim == 1:
        tensor = torch.from_numpy(data).unsqueeze(0)
    else:
        tensor = torch.from_numpy(data.T)
    return tensor, sr

import torchaudio
torchaudio.load = _soundfile_load

import transformers.pytorch_utils
if not hasattr(transformers.pytorch_utils, 'isin_mps_friendly'):
    def isin_mps_friendly(elements, test_elements):
        return torch.isin(elements, test_elements)
    transformers.pytorch_utils.isin_mps_friendly = isin_mps_friendly

# Lock to ensure thread-safety during TTS inference
tts_lock = threading.Lock()

# Paths to fine-tuned XTTS model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "functions", "xtts_fine-tuned")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
REF_AUDIO = os.path.join(MODEL_DIR, "stanley4.wav")

_tts_instance = None

def get_tts_model():
    global _tts_instance
    if _tts_instance is None:
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if os.path.exists(MODEL_DIR) and os.path.exists(CONFIG_PATH):
            print(f"[AUDIO GEN] Loading Fine-Tuned Stanley XTTS-v2 model on {device}...")
            _tts_instance = TTS(model_path=MODEL_DIR, config_path=CONFIG_PATH, progress_bar=False).to(device)
        else:
            print(f"[AUDIO GEN] Fine-tuned model not found at {MODEL_DIR}. Loading base XTTS-v2 model...")
            _tts_instance = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
    return _tts_instance

def get_audio(text: str, output_file: str, lang: str = 'fr'):
    """
    Generates a WAV audio file from text using fine-tuned Stanley XTTS-v2 model.
    """
    clean_text = text
    if clean_text.startswith("fr:"):
        clean_text = clean_text[3:].strip()
    elif clean_text.startswith("en:"):
        clean_text = clean_text[3:].strip()

    print(f"[AUDIO GEN] Synthesizing speech for Stanley ({lang}): '{clean_text[:60]}...'")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    with tts_lock:
        try:
            tts = get_tts_model()
            ref_wav = REF_AUDIO if os.path.exists(REF_AUDIO) else os.path.join(BASE_DIR, "dataset_french", "wavs", "stanley_0001.wav")

            tts.tts_to_file(
                text=clean_text,
                speaker_wav=ref_wav,
                language="fr",
                file_path=output_file,
                temperature=0.75,
                repetition_penalty=5.0,
                top_k=50,
                top_p=0.85,
                speed=1.0,
                enable_text_splitting=False
            )
            print(f"[AUDIO GEN] Successfully generated fine-tuned speech -> {output_file}")
            return output_file
        except Exception as e:
            print(f"[AUDIO GEN ERROR] Failed to generate speech: {e}")
            raise e