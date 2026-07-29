import os
import glob
import sys
import argparse
import static_ffmpeg
static_ffmpeg.add_paths()

import torch
import soundfile as sf
import librosa
import numpy as np
import whisper
from misaki import espeak
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Import kokoro symbol cleaner from kikiri-tts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kikiri-tts")))
try:
    from kikiri_tts.kokoro_symbols import symbols, dicts, TextCleaner
    cleaner = TextCleaner()
except Exception:
    cleaner = None

def preprocess_french_dataset(audio_dir: str, output_dir: str, sample_limit: int = None, whisper_model_size: str = "base"):
    os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)
    
    print(f"[DATASET PREP] Scanning source audio in: {audio_dir}")
    files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if not files:
        print(f"❌ No WAV files found in {audio_dir}")
        return

    if sample_limit:
        files = files[:sample_limit]
        print(f"[DATASET PREP] Limiting to first {sample_limit} files for test run")

    print(f"[DATASET PREP] Found {len(files)} source audio files.")
    print(f"[DATASET PREP] Loading Whisper ({whisper_model_size}) model...")
    whisper_model = whisper.load_model(whisper_model_size)

    print(f"[DATASET PREP] Initializing French G2P (fr-fr)...")
    g2p = espeak.EspeakG2P(language="fr-fr")

    manifest_entries = []
    chunk_counter = 1

    for file_idx, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        print(f"\n[DATASET PREP] [{file_idx+1}/{len(files)}] Processing '{fname}'...")
        try:
            sound = AudioSegment.from_file(fpath)
            
            # Split audio on silence into 3-12 second chunks
            chunks = split_on_silence(
                sound,
                min_silence_len=500,
                silence_thresh=sound.dBFS - 14,
                keep_silence=200
            )

            # Fallback if no silence detected or chunk too small
            if not chunks:
                chunks = [sound]

            # Merge very small chunks (< 2.0s) and split huge chunks (> 15.0s)
            merged_chunks = []
            cur = None
            for c in chunks:
                if len(c) < 1500: # skip under 1.5s
                    continue
                if cur is None:
                    cur = c
                elif len(cur) + len(c) < 10000:
                    cur += c
                else:
                    merged_chunks.append(cur)
                    cur = c
            if cur is not None and len(cur) >= 1500:
                merged_chunks.append(cur)

            if not merged_chunks:
                merged_chunks = [sound]

            print(f"  └ Sliced into {len(merged_chunks)} speech segments")

            for sub_idx, chunk_audio in enumerate(merged_chunks):
                chunk_id = f"stanley_{chunk_counter:04d}"
                out_wav_path = os.path.join(output_dir, "wavs", f"{chunk_id}.wav")

                # Export temp wav for Whisper & resample
                temp_wav = os.path.join(output_dir, "wavs", f"temp_{chunk_id}.wav")
                chunk_audio.export(temp_wav, format="wav")

                # Resample to 24000Hz mono PCM WAV (Kokoro standard)
                y, sr = librosa.load(temp_wav, sr=24000, mono=True)
                sf.write(out_wav_path, y, 24000, subtype='PCM_16')
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

                # Transcribe with Whisper (language="fr")
                res = whisper_model.transcribe(out_wav_path, language="fr")
                text = res.get("text", "").strip()

                if not text or len(text) < 3:
                    print(f"  ⚠️ Skip segment {chunk_id}: empty transcript")
                    continue

                # Phonemize French text
                phonemes, _ = g2p(text)
                phonemes = phonemes.strip()

                manifest_entries.append({
                    "wav_path": out_wav_path,
                    "text": text,
                    "phonemes": phonemes,
                    "speaker": "0"
                })

                print(f"  └ Segment {chunk_id} ({len(chunk_audio)/1000.0:.1f}s): \"{text[:60]}\" -> [{phonemes[:60]}]")
                chunk_counter += 1

        except Exception as e:
            print(f"  ❌ Error processing {fname}: {e}")

    print(f"\n[DATASET PREP SUCCESS] Processed {len(manifest_entries)} total segments.")

    # Split into train (90%) and val (10%)
    np.random.seed(42)
    indices = np.random.permutation(len(manifest_entries))
    val_count = max(1, int(len(manifest_entries) * 0.1))
    
    val_indices = set(indices[:val_count])
    train_entries = [e for i, e in enumerate(manifest_entries) if i not in val_indices]
    val_entries = [e for i, e in enumerate(manifest_entries) if i in val_indices]

    train_txt_path = os.path.join(output_dir, "train_list.txt")
    val_txt_path = os.path.join(output_dir, "val_list.txt")

    with open(train_txt_path, "w", encoding="utf-8") as f:
        for e in train_entries:
            f.write(f"{e['wav_path']}|{e['phonemes']}|0\n")

    with open(val_txt_path, "w", encoding="utf-8") as f:
        for e in val_entries:
            f.write(f"{e['wav_path']}|{e['phonemes']}|0\n")

    print(f"Saved {len(train_entries)} train samples -> {train_txt_path}")
    print(f"Saved {len(val_entries)} val samples -> {val_txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare French Kokoro dataset from audio folder")
    parser.add_argument("--audio-dir", type=str, default="/home/alexis/Downloads/stanley12")
    parser.add_argument("--output-dir", type=str, default="/home/alexis/Documents/projets/smart-radio-api/dataset_french")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit number of source files to process for testing")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model size: tiny, base, small, medium, large-v3")
    args = parser.parse_args()

    preprocess_french_dataset(args.audio_dir, args.output_dir, sample_limit=args.sample_limit, whisper_model_size=args.whisper_model)
