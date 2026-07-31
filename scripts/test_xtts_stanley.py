import os
import argparse
import torch
from pathlib import Path

# Fix compatibility between modern transformers and Coqui TTS
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
            tts = TTS(model_path=checkpoint_path, config_path=config_path, progress_bar=False).to(device)
        else:
            if checkpoint_path:
                print(f"⚠️ Checkpoint file '{checkpoint_path}' not found locally. Loading base XTTS-v2 model...")
            else:
                print("[1/2] Loading XTTS-v2 multilingual model...")
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)

        print(f"[2/2] Generating audio for text: '{text_to_speak[:60]}...'")
        tts.tts_to_file(
            text=text_to_speak,
            speaker_wav=ref_audio,
            language=language,
            file_path=output_path
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
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    generate_xtts_speech(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        ref_audio=args.ref_audio,
        text_to_speak=args.text,
        output_path=args.output,
        language=args.language,
        device=args.device
    )
