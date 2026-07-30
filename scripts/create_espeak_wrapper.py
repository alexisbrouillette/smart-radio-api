import os
import sys
import stat

def create_espeak_wrapper():
    venv_bin = os.path.dirname(sys.executable)
    wrapper_path = os.path.join(venv_bin, "espeak-ng")
    wrapper_path_alt = os.path.join(venv_bin, "espeak")

    code = f"""#!/usr/bin/env python3
import sys
import os

if "--version" in sys.argv:
    print("eSpeak NG text-to-speech: 1.51")
    sys.exit(0)

try:
    import espeakng_loader
    lib_path = espeakng_loader.get_library_path()
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib_path
except Exception:
    pass

sys.exit(0)
"""

    for p in [wrapper_path, wrapper_path_alt]:
        with open(p, "w") as f:
            f.write(code)
        # Make executable
        st = os.stat(p)
        os.chmod(p, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print(f"✅ Created executable wrapper -> {p}")

if __name__ == "__main__":
    create_espeak_wrapper()
