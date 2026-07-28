import sys
import time
import urllib.request

STREAM_URL = "https://alexisbrouillette--smart-radio-api-fastapi-app.modal.run/stream/live.mp3?track=Single%20Handed%20Sailor%20Dire%20Straits"

def test_live_stream_duration():
    print(f"\n=======================================================")
    print(f"🚀 AUTOMATED STREAM DURATION TEST")
    print(f"Target URL: {STREAM_URL}")
    print(f"=======================================================\n")
    
    start_time = time.time()
    total_bytes = 0
    chunks_count = 0

    try:
        req = urllib.request.Request(STREAM_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30.0) as response:
            print(f"[TEST] HTTP Status: {response.status}")
            print(f"[TEST] Content-Type: {response.headers.get('content-type')}")
            assert response.status == 200, f"Expected 200 OK, got {response.status}"
            
            while True:
                chunk = response.read(8192)
                if not chunk:
                    print(f"[TEST] Stream EOF reached naturally!")
                    break

                elapsed = time.time() - start_time
                total_bytes += len(chunk)
                chunks_count += 1

                if chunks_count % 50 == 0 or elapsed >= 15.0:
                    print(f"[TEST] Stream Elapsed: {elapsed:.2f}s | Chunks: {chunks_count} | Bytes: {total_bytes} ({total_bytes / 1024:.1f} KB)")

                # Test for 15 full seconds of uninterrupted streaming
                if elapsed >= 15.0:
                    print(f"\n[TEST] 15 Seconds Reached! Stopping active stream listener.")
                    break

    except Exception as e:
        print(f"❌ STREAM ERROR: {e}")
        sys.exit(1)

    elapsed_total = time.time() - start_time
    print(f"\n-------------------------------------------------------")
    print(f"📊 SUMMARY RESULTS:")
    print(f"Total Stream Duration Tested: {elapsed_total:.2f} seconds")
    print(f"Total Audio Chunks Received: {chunks_count}")
    print(f"Total Audio Data Received: {total_bytes} bytes ({total_bytes / 1024:.1f} KB)")
    print(f"Average Bitrate: {(total_bytes * 8 / elapsed_total) / 1000:.1f} kbps")
    print(f"-------------------------------------------------------")

    if elapsed_total < 14.0 or total_bytes < 100000:
        print(f"\n❌ FAIL: Stream stopped prematurely after {elapsed_total:.2f}s ({total_bytes} bytes)!")
        sys.exit(1)
    else:
        print(f"\n✅ PASS: Stream is continuously broadcasting without truncation!\n")

if __name__ == "__main__":
    test_live_stream_duration()
