import os
import sys
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

# Ensure kikiri-tts and kokoro paths are added
REPO_ROOT = Path(__file__).resolve().parents[1]
KIKIRI_DIR = REPO_ROOT / "kikiri-tts"
KOKORO_SUBMODULE = KIKIRI_DIR / "kokoro"

if KOKORO_SUBMODULE.exists() and str(KOKORO_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(KOKORO_SUBMODULE))
if str(KIKIRI_DIR) not in sys.path:
    sys.path.insert(0, str(KIKIRI_DIR))

def test_stanley_french_speech(
    checkpoint_path: str = "kikiri-tts/StyleTTS2/logs/kokoro-french-stanley/epoch_00009.pth",
    text_to_speak: str = "Bonjour! Vous écoutez Smart Radio avec la voix de Stanley. Bienvenue à tous nos auditeurs!",
    output_path: str = "test_output/stanley_french_sample.wav"
):
    print("=== 🎙️ TESTING FRENCH KOKORO TTS (STANLEY VOICE) ===")

    # 1. Check Checkpoint
    if not os.path.exists(checkpoint_path):
        # Look for latest epoch checkpoint in logs directory
        log_dir = "kikiri-tts/StyleTTS2/logs/kokoro-french-stanley"
        if os.path.exists(log_dir):
            ckpts = sorted([os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".pth")])
            if ckpts:
                checkpoint_path = ckpts[-1]
                print(f"ℹ️ Found latest checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"❌ No trained checkpoint found at '{checkpoint_path}'. Make sure Stage 1 training has saved at least 1 epoch!")
        return

    # 2. Extract Voicepack if not already present
    voice_dir = Path("voices")
    voice_dir.mkdir(exist_ok=True)
    voicepack_path = voice_dir / "stanley_french.pt"

    print(f"[1/3] Extracting Stanley voicepack to {voicepack_path}...")
    try:
        from scripts.extract_voicepack import extract_voicepack
        extract_voicepack(
            model_path=checkpoint_path,
            audio_dir="dataset_french/wavs",
            output_path=str(voicepack_path),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    except Exception as e:
        print(f"⚠️ Voicepack extraction note: {e}")

    # 3. Load Kokoro KPipeline & Voicepack
    print(f"[2/3] Loading Kokoro KPipeline (French fr-fr)...")
    try:
        from kokoro import KPipeline
        # French pipeline (lang_code='f')
        pipeline = KPipeline(lang_code="f", repo_id="hexgrad/Kokoro-82M")
        
        voice = torch.load(voicepack_path, map_location="cpu", weights_only=True) if voicepack_path.exists() else None

        print(f"[3/3] Synthesizing French speech...")
        print(f"  🗣️ Text: \"{text_to_speak}\"")

        generator = pipeline(text_to_speak, voice=voice, speed=1.0)
        audio_chunks = []
        for gs, ps, audio in generator:
            print(f"  └ IPA Phonemes: [{ps}]")
            audio_chunks.append(audio)

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, full_audio, 24000)
            print(f"\n🎉 [SUCCESS] Audio generated successfully!")
            print(f"  └ Saved to: {os.path.abspath(output_path)} ({len(full_audio)/24000:.2f}s)")
        else:
            print("⚠️ No audio output returned by generator.")

    except Exception as e:
        print(f"❌ Error during speech synthesis: {e}")

if __name__ == "__main__":
    test_stanley_french_speech()
