import os
import sys
import ctypes
import ctypes.util
import types
import torch
import soundfile as sf
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

# Ensure paths
REPO_ROOT = Path(__file__).resolve().parents[1]
KIKIRI_DIR = REPO_ROOT / "kikiri-tts" / "StyleTTS2"
KOKORO_SUBMODULE = KIKIRI_DIR / "kokoro"

if str(KIKIRI_DIR) not in sys.path:
    sys.path.insert(0, str(KIKIRI_DIR))
if KOKORO_SUBMODULE.exists() and str(KOKORO_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(KOKORO_SUBMODULE))

def test_direct_neural_inference(
    text_to_speak: str = "Bonjour et bienvenue sur Smart Radio! C'est Stanley, votre animateur en direct. Aujourd'hui, nous avons un programme musical exceptionnel avec le meilleur du jazz, de la soul et des grands classiques. Restez bien avec nous, la musique continue tout de suite!",
    output_path: str = "test_output/stanley_french_direct_stage2.wav"
):
    # Change CWD to StyleTTS2 directory for relative PLBERT configs
    os.chdir(KIKIRI_DIR)
    
    print("=== 🎙️ DIRECT FINE-TUNED NEURAL NETWORK INFERENCE (STAGE 2) ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  └ Using device: {device}")

    log_dir = REPO_ROOT / "kikiri-tts" / "StyleTTS2" / "logs" / "kokoro-french-stanley"
    second_stage_ckpt = log_dir / "second_stage.pth"

    if not second_stage_ckpt.exists():
        stage2_ckpts = sorted([log_dir / f for f in os.listdir(log_dir) if "2nd" in f and f.endswith(".pth")])
        if stage2_ckpts:
            second_stage_ckpt = stage2_ckpts[-1]

    if not second_stage_ckpt.exists():
        print(f"❌ Could not find Stage 2 checkpoint in {log_dir}")
        return

    print(f"  └ Loading fine-tuned model from: {second_stage_ckpt}")

    # Load configuration
    from models import build_model
    from utils import recursive_munch
    from Utils.PLBERT.util import load_plbert
    import yaml

    config_path = REPO_ROOT / "configs" / "config_french_stanley.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load PLBERT
    BERT_path = config.get("PLBERT_dir", False)
    plbert = load_plbert(BERT_path)

    # Build StyleTTS2 / Kokoro model structure
    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, None, None, plbert)

    # Load Stage 2 checkpoint weights
    checkpoint = torch.load(second_stage_ckpt, map_location=device)
    if "net" in checkpoint:
        for key in model:
            if key in checkpoint["net"] and model[key] is not None:
                model[key].load_state_dict(checkpoint["net"][key], strict=False)

    for key in model:
        if model[key] is not None and hasattr(model[key], "to"):
            model[key] = model[key].to(device).eval()

    # French G2P
    from misaki import espeak
    from kokoro_tb_utils import extract_voicepack, run_kokoro_inference

    g2p = espeak.EspeakG2P(language="fr-fr")
    phonemes, _ = g2p(text_to_speak)
    phonemes = phonemes.strip()

    print(f"  🗣️ Text: \"{text_to_speak}\"")
    print(f"  └ IPA Phonemes: [{phonemes}]")

    # Map text cleaner (standard Kokoro vocab)
    from kokoro_symbols import TextCleaner
    cleaner = TextCleaner()
    token_ids = cleaner(phonemes)

    # Extract Stage 2 mini-voicepack
    voicepack, ac_norm, pr_norm = extract_voicepack(model, str(REPO_ROOT / "dataset_french" / "wavs"), device, n_samples=200)
    print(f"  └ Extracted Stage 2 voicepack (Acoustic norm: {ac_norm:.4f}, Prosodic norm: {pr_norm:.4f})")

    # Run direct neural network inference
    # Apply duration scaling so speech is at natural 1.0x speed (~10s) instead of 15x stretched
    with torch.no_grad():
        ref_acoustic = voicepack[:128].unsqueeze(0)
        ref_prosodic = voicepack[128:].unsqueeze(0)

        input_ids = torch.LongTensor([[0, *token_ids, 0]]).to(device)
        input_lengths = torch.LongTensor([input_ids.shape[-1]]).to(device)
        text_mask = torch.gt(
            torch.arange(input_lengths.max()).unsqueeze(0).expand(1, -1).type_as(input_lengths) + 1,
            input_lengths.unsqueeze(1),
        ).to(device)

        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_prosodic = ref_prosodic
        d = model.predictor.text_encoder(d_en, s_prosodic, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)

        # Scale duration to natural 1.0x pace (approx 10-12s audio duration)
        pred_dur = torch.round((duration.squeeze() / 12.0)).clamp(min=1).long()
        if pred_dur.dim() == 0:
            pred_dur = pred_dur.unsqueeze(0)

        n_tokens = input_ids.shape[1]
        total_frames = int(pred_dur.sum().item())
        pred_aln_trg = torch.zeros(n_tokens, total_frames).to(device)
        c_frame = 0
        for i in range(n_tokens):
            dur_i = int(pred_dur[i].item())
            pred_aln_trg[i, c_frame : c_frame + dur_i] = 1
            c_frame += dur_i
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s_prosodic)

        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio_tensor = model.decoder(asr, F0_pred, N_pred, ref_acoustic)
        audio = audio_tensor.squeeze().cpu().numpy()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, 24000)
        print(f"\n🎉 [SUCCESS] Direct neural network audio generated successfully!")
        print(f"  └ Saved to: {os.path.abspath(output_path)} ({len(audio)/24000:.2f}s)")

if __name__ == "__main__":
    test_direct_neural_inference()
