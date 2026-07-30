import os
import sys
import tarfile
import urllib.request
from pathlib import Path

def install_espeak_binary():
    print("=== 📦 DOWNLOADING & UNPACKING ESPEAK-NG INTO .VENV ===")

    # Get .venv directory
    venv_dir = Path(sys.executable).parent.parent
    bin_dir = venv_dir / "bin"
    target_binary = bin_dir / "espeak-ng"

    if target_binary.exists():
        print(f"✅ espeak-ng binary already exists at: {target_binary}")
        return

    # Direct conda-forge linux-64 package URL for espeak-ng
    url = "https://anaconda.org/conda-forge/espeak-ng/1.51/download/linux-64/espeak-ng-1.51-h4e0d66e_0.tar.bz2"
    archive_path = venv_dir / "espeak_ng.tar.bz2"

    print(f"[1/2] Downloading espeak-ng binary package from Conda-Forge...")
    print(f"  └ URL: {url}")
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as response, open(archive_path, 'wb') as out_file:
        out_file.write(response.read())

    print(f"[2/2] Extracting binaries to {venv_dir}...")
    with tarfile.open(archive_path, 'r:bz2') as tar:
        tar.extractall(path=venv_dir)

    if archive_path.exists():
        archive_path.unlink()

    # Symlink espeak if needed
    espeak_symlink = bin_dir / "espeak"
    if not espeak_symlink.exists() and target_binary.exists():
        os.symlink("espeak-ng", espeak_symlink)

    print("🎉 [SUCCESS] espeak-ng installed successfully into .venv!")
    os.system(f"{target_binary} --version")

if __name__ == "__main__":
    install_espeak_binary()
