import os
import glob
import argparse
import numpy as np

# Import static_ffmpeg to ensure ffmpeg binaries are in PATH
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

import whisper

def prepare_xtts_dataset(
    audio_dir: str = "dataset_french/wavs",
    output_dir: str = "dataset_french",
    whisper_model_name: str = "base"
):
    train_csv = os.path.join(output_dir, "metadata_train.csv")
    val_csv = os.path.join(output_dir, "metadata_val.csv")

    if os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.getsize(train_csv) > 100:
        print(f"✅ XTTS metadata files already exist in '{output_dir}'. Skipping Whisper transcription!")
        return

    print("=== 🎙️ PREPARING XTTS-v2 DATASET ===")
    print(f"ℹ️ Scanning WAV files in: {audio_dir}")
    
    files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if not files:
        print(f"❌ No WAV files found in {audio_dir}")
        return

    print(f"ℹ️ Found {len(files)} WAV files.")
    print(f"[1/2] Transcribing with Whisper ({whisper_model_name})...")
    
    whisper_model = whisper.load_model(whisper_model_name)
    
    metadata_entries = []
    for idx, fpath in enumerate(files):
        try:
            res = whisper_model.transcribe(fpath, language="fr")
            text = res.get("text", "").strip()
            if text and len(text) >= 3:
                # Format: rel_audio_file|text|speaker_name (e.g. wavs/stanley_0001.wav|text|stanley)
                rel_fpath = os.path.relpath(fpath, output_dir)
                metadata_entries.append(f"{rel_fpath}|{text}|stanley")
                if (idx + 1) % 25 == 0 or (idx + 1) == len(files):
                    print(f"  └ Transcribed {idx+1}/{len(files)} files... (Latest: '{text[:50]}...')")
        except Exception as e:
            print(f"  ⚠️ Error transcribing {fpath}: {e}")

    print(f"[2/2] Splitting into train and validation sets...")
    np.random.seed(42)
    indices = np.random.permutation(len(metadata_entries))
    val_count = max(1, int(len(metadata_entries) * 0.1))
    
    val_indices = set(indices[:val_count])
    train_entries = [e for i, e in enumerate(metadata_entries) if i not in val_indices]
    val_entries = [e for i, e in enumerate(metadata_entries) if i in val_indices]

    train_csv = os.path.join(output_dir, "metadata_train.csv")
    val_csv = os.path.join(output_dir, "metadata_val.csv")

    with open(train_csv, "w", encoding="utf-8") as f:
        for line in train_entries:
            f.write(f"{line}\n")

    with open(val_csv, "w", encoding="utf-8") as f:
        for line in val_entries:
            f.write(f"{line}\n")

    print(f"\n🎉 [SUCCESS] XTTS Dataset preparation complete!")
    print(f"  └ Train set: {len(train_entries)} samples -> {train_csv}")
    print(f"  └ Val set:   {len(val_entries)} samples -> {val_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare XTTS-v2 CSV metadata dataset")
    parser.add_argument("--audio-dir", type=str, default="dataset_french/wavs")
    parser.add_argument("--output-dir", type=str, default="dataset_french")
    parser.add_argument("--whisper-model", type=str, default="base")
    args = parser.parse_args()

    prepare_xtts_dataset(args.audio_dir, args.output_dir, args.whisper_model)
