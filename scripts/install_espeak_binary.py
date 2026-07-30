import os
import sys
import shutil
import tarfile
import urllib.request
from pathlib import Path

def install_espeak_binary():
    print("=== 📦 CHECKING & ENFORCING ESPEAK-NG BINARY ===")

    venv_dir = Path(sys.executable).parent.parent
    bin_dir = venv_dir / "bin"
    lib_dir = venv_dir / "lib"
    share_dir = venv_dir / "share"
    target_binary = bin_dir / "espeak-ng"

    # Always ensure PATH & LD_LIBRARY_PATH are set in the current process
    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    espeak_lib = lib_dir / "libespeak-ng.so"
    if espeak_lib.exists():
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(espeak_lib)

    if not target_binary.exists():
        url = "https://anaconda.org/conda-forge/espeak-ng/1.51/download/linux-64/espeak-ng-1.51-h4e0d66e_0.tar.bz2"
        archive_path = venv_dir / "espeak_ng.tar.bz2"

        print(f"[1/2] Downloading espeak-ng binary package from Conda-Forge...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req) as response, open(archive_path, 'wb') as out_file:
            out_file.write(response.read())

        print(f"[2/2] Extracting binaries to {venv_dir}...")
        with tarfile.open(archive_path, 'r:bz2') as tar:
            tar.extractall(path=venv_dir)

        if archive_path.exists():
            archive_path.unlink()

        # Symlink espeak if missing
        espeak_symlink = bin_dir / "espeak"
        if not espeak_symlink.exists() and target_binary.exists():
            try:
                os.symlink("espeak-ng", espeak_symlink)
            except Exception:
                pass

        # Make executable
        os.chmod(target_binary, 0o755)

    espeak_path = shutil.which("espeak-ng") or shutil.which("espeak")
    print(f"✅ Verified espeak executable in PATH: {espeak_path}")

if __name__ == "__main__":
    install_espeak_binary()
