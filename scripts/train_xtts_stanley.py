import os
import sys
import argparse
from pathlib import Path

# Add Coqui-ai-TTS and project paths
REPO_ROOT = Path(__file__).resolve().parents[1]
COQUI_DIR = Path("/home/alexis/Documents/projets/coqui-ai-TTS")

if COQUI_DIR.exists() and str(COQUI_DIR) not in sys.path:
    sys.path.insert(0, str(COQUI_DIR))

# Fix compatibility between modern transformers and Coqui TTS
import soundfile as sf
import torch
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

def run_xtts_training(
    train_csv: str = "dataset_french/metadata_train.csv",
    eval_csv: str = "dataset_french/metadata_val.csv",
    output_dir: str = "functions/xtts_fine-tuned",
    epochs: int = 10,
    batch_size: int = 2,
    grad_accum: int = 2,
    language: str = "fr"
):
    print("=== 🚀 XTTS-v2 FINE-TUNING PIPELINE (STANLEY) ===")
    print(f"ℹ️ Train CSV:  {train_csv}")
    print(f"ℹ️ Eval CSV:   {eval_csv}")
    print(f"ℹ️ Output Dir: {output_dir}")
    print(f"ℹ️ Epochs:     {epochs}")
    print(f"ℹ️ Batch Size: {batch_size} (Grad Accum: {grad_accum})")

    # 1. Verify / Generate Metadata CSVs if missing
    if not os.path.exists(train_csv) or not os.path.exists(eval_csv):
        print("⚠️ Metadata CSV files missing. Running dataset preparation...")
        from scripts.prepare_xtts_dataset import prepare_xtts_dataset
        prepare_xtts_dataset(audio_dir="dataset_french/wavs", output_dir="dataset_french")

    # 2. Convert to absolute paths so Coqui's dataset loader resolves paths cleanly
    abs_train_csv = os.path.abspath(train_csv)
    abs_eval_csv = os.path.abspath(eval_csv)
    abs_output_dir = os.path.abspath(output_dir)

    os.makedirs(abs_output_dir, exist_ok=True)

    # 3. Invoke Coqui XTTS train_gpt
    try:
        from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt

        config_file, ckpt_file, vocab_file, trainer_out_path, speaker_ref = train_gpt(
            language=language,
            num_epochs=epochs,
            batch_size=batch_size,
            grad_acumm=grad_accum,
            train_csv=abs_train_csv,
            eval_csv=abs_eval_csv,
            output_path=abs_output_dir
        )

        print(f"\n🎉 [SUCCESS] XTTS Fine-Tuning Complete!")
        print(f"  └ Output Checkpoints saved to: {os.path.abspath(trainer_out_path)}")

    except BaseException as e:
        print(f"❌ Error during XTTS Fine-Tuning: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tune XTTS-v2 for Stanley Voice")
    parser.add_argument("--train-csv", type=str, default="dataset_french/metadata_train.csv", help="Path to train CSV metadata")
    parser.add_argument("--eval-csv", type=str, default="dataset_french/metadata_val.csv", help="Path to validation CSV metadata")
    parser.add_argument("--output-dir", type=str, default="functions/xtts_fine-tuned", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU step")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--language", type=str, default="fr", help="Language code (default: 'fr')")
    args = parser.parse_args()

    run_xtts_training(
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        language=args.language
    )
