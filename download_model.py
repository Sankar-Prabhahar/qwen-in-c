import urllib.request
import os
import sys
import time

url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q6_K.gguf"
dest = "tinyllama.gguf"

print(f"Downloading complete {dest} (904 MB)...")

def download_with_progress(url, dest):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(dest + ".tmp", 'wb') as f:
        total = int(resp.headers.get('Content-Length', 0))
        downloaded = 0
        start = time.time()
        last_print = start
        
        while True:
            chunk = resp.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            
            now = time.time()
            if now - last_print >= 2.0 or downloaded == total:
                speed_mb = (downloaded / (1024 * 1024)) / max(now - start, 0.001)
                pct = (downloaded / total) * 100.0 if total else 0
                print(f"  {downloaded / (1024*1024):.1f} / {total / (1024*1024):.1f} MB ({pct:.1f}%) - {speed_mb:.1f} MB/s")
                sys.stdout.flush()
                last_print = now

    if os.path.exists(dest):
        os.remove(dest)
    os.rename(dest + ".tmp", dest)
    print("Download complete!")

if __name__ == '__main__':
    download_with_progress(url, dest)
