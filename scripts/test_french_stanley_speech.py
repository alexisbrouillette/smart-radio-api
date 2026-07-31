import os
import sys
import ctypes
import ctypes.util
import types
import argparse
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

# ==============================================================================
# ESPEAK-NG INTERCEPT & MONKEYPATCH
# ==============================================================================
HOME = os.path.expanduser("~")
INSTALL_DIR = os.path.join(HOME, "espeak-ng-install")
ESPEAK_SO = os.path.join(INSTALL_DIR, "lib64", "libespeak-ng.so")

if not os.path.exists(ESPEAK_SO):
    ESPEAK_SO = os.path.join(INSTALL_DIR, "lib", "libespeak-ng.so")

if os.path.exists(ESPEAK_SO):
    print(f"✅ Intercepting TTS libraries to use: {ESPEAK_SO}")
    
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = ESPEAK_SO
    os.environ["ESPEAK_DATA_PATH"] = os.path.join(INSTALL_DIR, "share")
    os.environ["PATH"] = f"{os.path.join(INSTALL_DIR, 'bin')}:{os.environ.get('PATH', '')}"

    mock_loader = types.ModuleType("espeakng_loader")
    mock_loader.get_library_path = lambda: ESPEAK_SO
    mock_loader.get_data_path = lambda: os.path.join(INSTALL_DIR, "share", "espeak-ng-data")
    sys.modules["espeakng_loader"] = mock_loader

    _original_find_library = ctypes.util.find_library
    def _mock_find_library(name):
        if name in ['espeak-ng', 'espeak']:
            return ESPEAK_SO
        return _original_find_library(name)
    
    ctypes.util.find_library = _mock_find_library

# Ensure kikiri-tts and kokoro paths are added
REPO_ROOT = Path(__file__).resolve().parents[1]
KIKIRI_DIR = REPO_ROOT / "kikiri-tts"
KOKORO_SUBMODULE = KIKIRI_DIR / "kokoro"

if KOKORO_SUBMODULE.exists() and str(KOKORO_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(KOKORO_SUBMODULE))
if str(KIKIRI_DIR) not in sys.path:
    sys.path.insert(0, str(KIKIRI_DIR))

def test_stanley_french_speech(
    checkpoint_path: str = None,
    voicepack_path: str = "voices/stanley_french.pt",
    text_to_speak: str = "Bonjour et bienvenue sur Smart Radio! C'est Stanley, votre animateur en direct. Aujourd'hui, nous avons un programme musical exceptionnel avec le meilleur du jazz, de la soul et des grands classiques. Restez bien avec nous, la musique continue tout de suite!",
    output_path: str = "test_output/stanley_french_sample.wav",
    audio_dir: str = "dataset_french/wavs",
    speed: float = 1.0
):
    print("=== 🎙️ TESTING FRENCH KOKORO TTS (STANLEY VOICE) ===")

    # 1. Locate Checkpoint if not specified
    log_dir = "kikiri-tts/StyleTTS2/logs/kokoro-french-stanley"
    if not checkpoint_path and os.path.exists(log_dir):
        second_stage = os.path.join(log_dir, "second_stage.pth")
        stage2_ckpts = sorted([os.path.join(log_dir, f) for f in os.listdir(log_dir) if "2nd" in f and f.endswith(".pth")])
        if os.path.exists(second_stage):
            checkpoint_path = second_stage
        elif stage2_ckpts:
            checkpoint_path = stage2_ckpts[-1]
        else:
            ckpts = sorted([os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".pth")])
            if ckpts:
                checkpoint_path = ckpts[-1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ℹ️ Device: {device}")

    # 2. Extract Stage 2 Voicepack if checkpoint exists and voicepack missing or new checkpoint passed
    voicepack_file = Path(voicepack_path)
    voicepack_file.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"ℹ️ Extracting voicepack from checkpoint: {checkpoint_path}")
        try:
            from kokoro_tb_utils import extract_voicepack, build_kokoro_model
            
            # Load model and extract
            model = build_kokoro_model(checkpoint_path, device=device)
            voicepack, ac_norm, pr_norm = extract_voicepack(
                model=model,
                audio_dir=audio_dir if os.path.exists(audio_dir) else "dataset_french/wavs",
                device=device,
                n_samples=200
            )
            torch.save(voicepack, str(voicepack_file))
            print(f"  └ Saved voicepack to {voicepack_file} (Acoustic norm: {ac_norm:.4f}, Prosodic norm: {pr_norm:.4f})")
        except Exception as e:
            print(f"⚠️ Voicepack extraction note: {e}")
    
    # 3. Verify Voicepack Existence
    if not voicepack_file.exists():
        print(f"❌ Voicepack file not found at '{voicepack_file}' and no valid checkpoint found at '{checkpoint_path}'.")
        print("  └ Make sure you have trained a checkpoint or copied your 'stanley_french.pt' into 'voices/'.")
        return

    # 4. Load Kokoro French Pipeline & Synthesize
    print(f"[2/2] Synthesizing speech with Kokoro French Native Engine (lang_code='f')...")
    try:
        from kokoro import KPipeline

        # Load voicepack tensor or path
        print(f"  └ Loading voicepack from {voicepack_file}...")
        voice_input = str(voicepack_file) if voicepack_file.exists() else voicepack_path

        # Initialize Kokoro native French pipeline
        pipeline = KPipeline(lang_code='f', repo_id="hexgrad/Kokoro-82M")

        # Generate audio using Kokoro Native French Pipeline
        generator = pipeline(text_to_speak, voice=voice_input, speed=speed)
        
        audios = []
        for _, _, chunk in generator:
            if chunk is not None and len(chunk) > 0:
                audios.append(chunk)
        
        if audios:
            full_audio = np.concatenate(audios)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, full_audio, 24000)
            print(f"\n🎉 [SUCCESS] French Stanley Audio generated successfully!")
            print(f"  └ Saved to: {os.path.abspath(output_path)} ({len(full_audio)/24000:.2f}s)")
        else:
            print("⚠️ No audio output returned.")

    except Exception as e:
        print(f"❌ Error during speech synthesis: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test French Kokoro TTS Stanley Voice")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth checkpoint file")
    parser.add_argument("--voicepack", type=str, default="voices/stanley_french.pt", help="Path to voicepack .pt file")
    parser.add_argument("--text", type=str, default="Bonjour et bienvenue sur Smart Radio! C'est Stanley, votre animateur en direct. Aujourd'hui, nous avons un programme musical exceptionnel avec le meilleur du jazz, de la soul et des grands classiques. Restez bien avec nous, la musique continue tout de suite!", help="Text to speak")
    parser.add_argument("--output", type=str, default="test_output/stanley_french_sample.wav", help="Output WAV file path")
    parser.add_argument("--audio-dir", type=str, default="dataset_french/wavs", help="Path to reference audio dataset WAVs")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier")
    args = parser.parse_args()

    test_stanley_french_speech(
        checkpoint_path=args.checkpoint,
        voicepack_path=args.voicepack,
        text_to_speak=args.text,
        output_path=args.output,
        audio_dir=args.audio_dir,
        speed=args.speed
    )
