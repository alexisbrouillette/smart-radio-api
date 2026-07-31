import os
import argparse
import soundfile as sf
import torch
from pathlib import Path

# Add static_ffmpeg paths if available
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

# Patch torchaudio.load to use soundfile (bypasses torchcodec CUDA library errors)
import torchaudio

def _soundfile_load(filepath, **kwargs):
    data, sr = sf.read(filepath, dtype="float32")
    if data.ndim == 1:
        tensor = torch.from_numpy(data).unsqueeze(0)
    else:
        tensor = torch.from_numpy(data.T)
    return tensor, sr

torchaudio.load = _soundfile_load

DEFAULT_REF_TEXT = "Terminez-moi tu filles chanson originale de la chanteuse et pianiste Anna BaStow, qui figure sur son album parue en deux mille vingt-et-un."

def generate_f5_speech(
    ref_audio: str = "dataset_french/wavs/stanley_0001.wav",
    ref_text: str = DEFAULT_REF_TEXT,
    text_to_speak: str = "Bonjour et bienvenue sur Smart Radio! C'est Stanley en direct. Restez bien avec nous, le meilleur du jazz, de la soul et des grands classiques continue tout de suite!",
    output_path: str = "test_output/f5_stanley_sample.wav",
    device: str = None
):
    print("=== 🎙️ F5-TTS FRENCH VOICE CLONING (STANLEY) ===")
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ℹ️ Device: {device}")
    print(f"ℹ️ Reference Audio: {ref_audio}")

    if not os.path.exists(ref_audio):
        print(f"❌ Reference audio file not found at: {ref_audio}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        from f5_tts.api import F5TTS

        print("[1/2] Loading F5-TTS model pipeline...")
        f5tts = F5TTS(device=device)

        print(f"[2/2] Generating audio for text: '{text_to_speak[:60]}...'")
        wav, sr, _ = f5tts.infer(
            ref_file=ref_audio,
            ref_text=ref_text or DEFAULT_REF_TEXT,
            gen_text=text_to_speak
        )

        sf.write(output_path, wav, sr)
        print(f"\n🎉 [SUCCESS] F5-TTS Audio generated successfully!")
        print(f"  └ Saved to: {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"❌ Error during F5-TTS speech synthesis: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test F5-TTS Zero-Shot Voice Cloning for Stanley")
    parser.add_argument("--ref-audio", type=str, default="dataset_french/wavs/stanley_0001.wav", help="Path to 5-10s clean reference WAV file of Stanley")
    parser.add_argument("--ref-text", type=str, default=DEFAULT_REF_TEXT, help="Transcript of reference audio clip")
    parser.add_argument("--text", type=str, default="Bonjour et bienvenue sur Smart Radio! C'est Stanley en direct. Restez bien avec nous, le meilleur du jazz, de la soul et des grands classiques continue tout de suite!", help="Text to speak")
    parser.add_argument("--output", type=str, default="test_output/f5_stanley_sample.wav", help="Output WAV file path")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    generate_f5_speech(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        text_to_speak=args.text,
        output_path=args.output,
        device=args.device
    )
