import os
import argparse
import torch
from pathlib import Path

# Fix compatibility between modern transformers and Coqui TTS
import warnings
warnings.filterwarnings('ignore')

import logging
try:
    from transformers import logging as tf_logging
    tf_logging.set_verbosity_error()
except Exception:
    pass

import soundfile as sf
import torchaudio

def _soundfile_load(filepath, **kwargs):
    data, sr = sf.read(filepath, dtype="float32")
    if data.ndim == 1:
        tensor = torch.from_numpy(data).unsqueeze(0)
    else:
        tensor = torch.from_numpy(data.T)
    return tensor, sr

torchaudio.load = _soundfile_load

import transformers.pytorch_utils
if not hasattr(transformers.pytorch_utils, 'isin_mps_friendly'):
    def isin_mps_friendly(elements, test_elements):
        return torch.isin(elements, test_elements)
    transformers.pytorch_utils.isin_mps_friendly = isin_mps_friendly

def generate_xtts_speech(
    checkpoint_path: str = None,
    config_path: str = None,
    ref_audio: str = "functions/xtts_fine-tuned/stanley4.wav",
    text_to_speak: str = "Bienvenue sur Smart Radio! Je suis Stanley. La prochaine chanson est un morceau exceptionnel de jazz et de soul. Bonne écoute!",
    output_path: str = "test_output/xtts_stanley_radio_intro.wav",
    language: str = "fr",
    pause_sec: float = 0.0,
    device: str = None
):
    print("=== 🎙️ XTTS-v2 FRENCH SPEECH SYNTHESIS (STANLEY) ===")
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ℹ️ Device: {device}")

    # Use dataset clip fallback if stanley4.wav doesn't exist
    if not os.path.exists(ref_audio):
        fallback_ref = "dataset_french/wavs/stanley_0001.wav"
        if os.path.exists(fallback_ref):
            ref_audio = fallback_ref

    print(f"ℹ️ Reference Audio: {ref_audio}")
    print(f"ℹ️ Language: {language}")
    print(f"ℹ️ Text: '{text_to_speak}'")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        from TTS.api import TTS

        if checkpoint_path and os.path.exists(checkpoint_path) and config_path and os.path.exists(config_path):
            print(f"[1/2] Loading FINE-TUNED XTTS model from: {checkpoint_path}...")
            model_dir = os.path.dirname(checkpoint_path) if os.path.isfile(checkpoint_path) else checkpoint_path
            
            # Ensure model.pth exists in model_dir for Coqui TTS API
            target_model_pth = os.path.join(model_dir, "model.pth")
            if not os.path.exists(target_model_pth) and os.path.isfile(checkpoint_path):
                import shutil
                print(f"ℹ️ Copying '{os.path.basename(checkpoint_path)}' to 'model.pth'...")
                shutil.copyfile(checkpoint_path, target_model_pth)

            tts = TTS(model_path=model_dir, config_path=config_path, progress_bar=False).to(device)
        else:
            if checkpoint_path:
                print(f"⚠️ Checkpoint file '{checkpoint_path}' not found locally. Loading base XTTS-v2 model...")
            else:
                print("[1/2] Loading XTTS-v2 multilingual model...")
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)

        print(f"[2/2] Generating smooth fine-tuned audio with {pause_sec}s sentence pauses...")
        import re
        import numpy as np

        # Split text into distinct sentences at punctuation marks (. ! ?)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_to_speak) if s.strip()]
        
        if len(sentences) > 1 and pause_sec > 0:
            print(f"ℹ️ Split text into {len(sentences)} distinct radio phrases with {pause_sec}s pause padding.")
            temp_files = []
            audio_chunks = []
            sample_rate = 24000
            silence_samples = np.zeros(int(sample_rate * pause_sec), dtype=np.float32)

            for i, sentence in enumerate(sentences):
                temp_wav = f"{output_path}_part_{i}.wav"
                temp_files.append(temp_wav)
                tts.tts_to_file(
                    text=sentence,
                    speaker_wav=ref_audio,
                    language=language,
                    file_path=temp_wav,
                    temperature=0.70,
                    repetition_penalty=6.0,
                    top_k=50,
                    top_p=0.85,
                    speed=1.0
                )
                data, sr = sf.read(temp_wav, dtype="float32")
                sample_rate = sr
                audio_chunks.append(data)
                if i < len(sentences) - 1:
                    audio_chunks.append(silence_samples)

            # Concatenate all sentence audio chunks with explicit silence padding
            final_audio = np.concatenate(audio_chunks)
            sf.write(output_path, final_audio, sample_rate)

            # Cleanup temp part files
            for tf in temp_files:
                if os.path.exists(tf):
                    os.remove(tf)
        else:
            tts.tts_to_file(
                text=text_to_speak,
                speaker_wav=ref_audio,
                language=language,
                file_path=output_path,
                temperature=0.70,
                repetition_penalty=6.0,
                top_k=50,
                top_p=0.85,
                speed=1.0
            )

        print(f"\n🎉 [SUCCESS] XTTS-v2 Audio generated successfully!")
        print(f"  └ Saved to: {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"❌ Error during XTTS-v2 speech synthesis: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test XTTS-v2 Native French Voice Cloning for Stanley")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to fine-tuned .pth checkpoint file")
    parser.add_argument("--config", type=str, default="functions/xtts_fine-tuned/config.json", help="Path to fine-tuned config.json file")
    parser.add_argument("--ref-audio", type=str, default="functions/xtts_fine-tuned/stanley4.wav", help="Path to reference WAV file of Stanley")
    parser.add_argument("--text", type=str, default="Bienvenue sur Smart Radio! Je suis Stanley. La prochaine chanson est un morceau exceptionnel de jazz et de soul. Bonne écoute!", help="Text to speak")
    parser.add_argument("--output", type=str, default="test_output/xtts_stanley_radio_intro.wav", help="Output WAV file path")
    parser.add_argument("--language", type=str, default="fr", help="Language code (default: 'fr')")
    parser.add_argument("--pause-sec", type=float, default=0.0, help="Silence pause duration in seconds between sentences (default: 0.0 for continuous natural speech)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    generate_xtts_speech(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        ref_audio=args.ref_audio,
        text_to_speak=args.text,
        output_path=args.output,
        language=args.language,
        pause_sec=args.pause_sec,
        device=args.device
    )
