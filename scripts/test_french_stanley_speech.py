import os
import sys
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

# Ensure EspeakWrapper & espeakng_loader compatibility
try:
    import espeakng_loader
    if hasattr(espeakng_loader, 'make_library_available'):
        espeakng_loader.make_library_available()
    lib_path = espeakng_loader.get_library_path()
    data_path = espeakng_loader.get_data_path()
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib_path
    os.environ["ESPEAK_DATA_PATH"] = data_path
    os.environ["PHONEMIZER_ESPEAK_PATH"] = os.path.dirname(lib_path)
except Exception as e:
    pass

try:
    import phonemizer
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.is_available = staticmethod(lambda: True)
    if 'lib_path' in locals():
        EspeakWrapper._ESPEAK_LIBRARY = lib_path
        try:
            EspeakWrapper.set_library_path(lib_path)
        except Exception:
            pass
    if not hasattr(EspeakWrapper, 'set_data_path'):
        EspeakWrapper.set_data_path = staticmethod(lambda *args, **kwargs: None)
except Exception:
    pass

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
    text_to_speak: str = "Bonjour et bienvenue sur Smart Radio! C'est Stanley, votre animateur en direct. Aujourd'hui, nous avons un programme musical exceptionnel avec le meilleur du jazz, de la soul et des grands classiques. Restez bien avec nous, la musique continue tout de suite!",
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

    # 3. Load Kokoro KModel & French G2P
    print(f"[2/3] Loading Kokoro KModel and French G2P...")
    try:
        from kokoro import KModel
        from misaki import espeak
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  └ Using device: {device}")

        # Initialize KModel
        kmodel = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()

        # Load extracted voicepack
        print(f"  └ Loading voicepack from {voicepack_path}...")
        voice = torch.load(voicepack_path, map_location=device, weights_only=True)

        # Phonemize French text
        try:
            g2p = espeak.EspeakG2P(language="fr-fr")
            phonemes, _ = g2p(text_to_speak)
        except Exception as g2p_err:
            print(f"  ⚠️ misaki G2P note ({g2p_err}), using phonemizer fallback...")
            import phonemizer
            phonemes = phonemizer.phonemize(text_to_speak, language="fr-fr", backend="espeak", strip=True)

        phonemes = phonemes.strip()
        print(f"  └ IPA Phonemes: [{phonemes}]")

        if len(phonemes) > 510:
            phonemes = phonemes[:510]

        # Generate audio using KModel directly
        output = kmodel(phonemes, voice[len(phonemes)-1], 1.0, return_output=True)
        audio = output.audio.cpu().numpy()

        if len(audio) > 0:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio, 24000)
            print(f"\n🎉 [SUCCESS] Audio generated successfully!")
            print(f"  └ Saved to: {os.path.abspath(output_path)} ({len(audio)/24000:.2f}s)")
        else:
            print("⚠️ No audio output returned.")

    except Exception as e:
        print(f"❌ Error during speech synthesis: {e}")

if __name__ == "__main__":
    test_stanley_french_speech()
