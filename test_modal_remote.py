import modal

app = modal.App("test-stream-debug")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("yt-dlp", "fastapi", "httpx")
)

@app.function(image=image)
def test_pre_download_remote(track: str):
    import yt_dlp
    import subprocess
    import os

    logs = []
    logs.append(f"Testing SoundCloud pre_download for '{track}' inside Modal container...")
    
    url = None

    # SoundCloud Search
    try:
        ydl_opts_sc = {
            'format': 'bestaudio/best',
            'quiet': True,
            'no_warnings': True,
            'default_search': 'scsearch1'
        }
        with yt_dlp.YoutubeDL(ydl_opts_sc) as ydl:
            info = ydl.extract_info(track, download=False)
            if 'entries' in info and len(info['entries']) > 0 and info['entries'][0].get('url'):
                url = info['entries'][0]['url']
            elif info.get('url'):
                url = info['url']
            logs.append(f"SoundCloud URL -> {url[:100]}...")
    except Exception as e:
        logs.append(f"SoundCloud error: {e}")

    if not url:
        logs.append("❌ Could not get URL from SoundCloud!")
        return "\n".join(logs)

    dest_file = "/tmp/test_download.mp3"
    cmd = [
        'ffmpeg',
        '-y',
        '-reconnect', '1',
        '-reconnect_streamed', '1',
        '-reconnect_delay_max', '5',
        '-user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        '-i', url,
        '-vn',
        '-acodec', 'libmp3lame',
        '-b:a', '128k',
        '-ar', '44100',
        '-ac', '2',
        dest_file
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    logs.append(f"FFmpeg exit code: {proc.returncode}")
    
    if os.path.exists(dest_file):
        size = os.path.getsize(dest_file)
        logs.append(f"Downloaded SoundCloud file size: {size} bytes ({size / 1024:.1f} KB)")
    return "\n".join(logs)

if __name__ == "__main__":
    with app.run():
        res = test_pre_download_remote.remote("It's Too Late Carole King")
        print("\n--- MODAL LOGS ---")
        print(res)
