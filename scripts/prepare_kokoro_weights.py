import os
import sys
import torch
from huggingface_hub import hf_hub_download

def prepare_kokoro_weights(output_path: str = "kokoro_base.pth"):
    print("[WEIGHT CONVERT] Checking Kokoro-82M base weights...")
    raw_path = "kokoro-v1_0.pth"
    if not os.path.exists(raw_path):
        print("[WEIGHT CONVERT] Downloading kokoro-v1_0.pth from HuggingFace...")
        raw_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="kokoro-v1_0.pth")

    print(f"[WEIGHT CONVERT] Loading raw Kokoro weights from '{raw_path}'...")
    raw = torch.load(raw_path, weights_only=False, map_location="cpu")

    def strip_prefix(state_dict):
        if not isinstance(state_dict, dict):
            return state_dict
        return {k.replace('module.', ''): v for k, v in state_dict.items()}

    net = {
        'bert': strip_prefix(raw.get('bert', {})),
        'bert_encoder': strip_prefix(raw.get('bert_encoder', {})),
        'predictor': strip_prefix(raw.get('predictor', {})),
        'text_encoder': strip_prefix(raw.get('text_encoder', {})),
        'decoder': strip_prefix(raw.get('decoder', {})),
    }

    torch.save({'net': net}, output_path)
    print(f"[WEIGHT CONVERT SUCCESS] Saved StyleTTS2-compatible weights -> {output_path} ({os.path.getsize(output_path) / (1024*1024):.1f} MB)")

if __name__ == "__main__":
    prepare_kokoro_weights()
