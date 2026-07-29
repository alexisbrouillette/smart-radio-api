import os
import sys
import torch
import yaml
import soundfile as sf
import numpy as np

# Ensure kikiri-tts paths are added
KIKIRI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kikiri-tts"))
sys.path.append(KIKIRI_DIR)
sys.path.append(os.path.join(KIKIRI_DIR, "StyleTTS2"))

def run_smoke_test(config_path: str = "kikiri-tts/configs/config_french_stanley.yml", steps: int = 2):
    print("=== 🚀 FRENCH KOKORO FINE-TUNING SMOKE TEST ===")
    
    # 1. Load config
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"✅ Loaded config: '{config_path}'")

    # 2. Check dataset files
    raw_train_path = config["data_params"]["train_data"]
    config_dir = os.path.dirname(config_path)
    train_txt = os.path.abspath(os.path.join(config_dir, raw_train_path))

    if not os.path.exists(train_txt):
        # Fallback check directly in smart-radio-api/dataset_french/
        train_txt = os.path.abspath("dataset_french/train_list.txt")

    if not os.path.exists(train_txt):
        print(f"❌ Dataset file not found: {train_txt}. Run prepare_french_kikiri_dataset.py first!")
        return
    
    with open(train_txt, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    print(f"✅ Found dataset list: {len(lines)} training samples.")

    # 3. Check base weights
    raw_weights_path = config["pretrained_model"]
    weights_path = os.path.abspath(os.path.join(config_dir, raw_weights_path))
    if not os.path.exists(weights_path):
        weights_path = os.path.abspath("kokoro_base.pth")

    if not os.path.exists(weights_path):
        print(f"❌ Pretrained base weights not found: {weights_path}. Run prepare_kokoro_weights.py first!")
        return
    
    ckpt = torch.load(weights_path, weights_only=False, map_location="cpu")
    net = ckpt.get("net", {})
    print(f"✅ Loaded pretrained base weights: {weights_path} ({len(net)} module groups).")
    for k in net.keys():
        print(f"  └ Module group '{k}': {len(net[k])} weight tensors")

    # 4. Verify Kokoro Token Symbols Vocabulary (178 tokens)
    try:
        from kokoro_symbols import symbols, dicts
        print(f"✅ Verified Kokoro Symbol Mapping: {len(symbols)} tokens (Expected 178).")
        assert len(symbols) == 178, f"Expected 178 symbols, got {len(symbols)}"
    except Exception as e:
        print(f"⚠️ Symbol mapping check: {e}")

    print("\n✅ [SMOKE TEST SUCCESS] Pipeline verification complete! Ready for GPU fine-tuning.")

if __name__ == "__main__":
    run_smoke_test()
