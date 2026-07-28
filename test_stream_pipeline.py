import os
import ssl
import sys
import urllib.request
import pytest
from app.main import execute_gpu_task, get_cache_filepath

def test_kokoro_tts_mp3_conversion():
    """Test that Kokoro TTS generates a valid, non-empty MP3 audio file"""
    output_mp3 = "test_unit_speech.mp3"
    if os.path.exists(output_mp3):
        os.remove(output_mp3)
        
    result_path = execute_gpu_task("That was Paul Simon! Up next, Carole King!", output_mp3)
    
    assert os.path.exists(result_path), "Kokoro TTS output MP3 file does not exist!"
    assert result_path.endswith(".mp3"), f"Expected .mp3 output file, got {result_path}"
    
    file_size = os.path.getsize(result_path)
    print(f"\n[UNIT TEST PASS] Kokoro TTS MP3 Generated: {result_path} ({file_size} bytes)")
    assert file_size > 15000, f"MP3 file size too small: {file_size} bytes"

    # Clean up test output
    if os.path.exists(output_mp3):
        os.remove(output_mp3)

def test_cache_filepath_hashing():
    """Test deterministic cache path generation"""
    path1 = get_cache_filepath("It's Too Late Carole King")
    path2 = get_cache_filepath("It's Too Late Carole King")
    assert path1 == path2, "Cache filepath hashing must be deterministic!"

def test_live_stream_endpoint_response():
    """Test live broadcast stream endpoint returning 200 OK and valid MP3 byte chunks"""
    url = "https://127.0.0.1:8000/stream/live.mp3?track=Here%20America&hostText=Up%20next%2C%20Carole%20King!"
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url)
    
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
            assert response.status == 200, f"Expected 200 OK, got {response.status}"
            assert response.headers.get("Content-Type") == "audio/mpeg", "Expected audio/mpeg header"
            
            # Read 32KB of broadcast audio bytes
            chunk = response.read(32768)
            assert len(chunk) == 32768, f"Expected 32768 bytes, read {len(chunk)}"
            print(f"\n[INTEGRATION TEST PASS] Live stream endpoint returned HTTP 200 with 32KB audio chunks!")
    except Exception as e:
        pytest.fail(f"Live stream endpoint test failed: {e}")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
