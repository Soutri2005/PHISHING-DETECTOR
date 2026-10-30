"""
Dataset downloader script.
Downloads the real dataset from Hugging Face into data/raw/Phishing_Email.csv.
"""

import os
import sys
import urllib.request
import time

URLS = [
    "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/resolve/main/Phishing_Email.csv",
    "https://raw.githubusercontent.com/zefang-liu/phishing-email-dataset/main/Phishing_Email.csv",
]

def download_data(output_path: str) -> bool:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000000:
        print(f"Dataset already exists at: {output_path} ({os.path.getsize(output_path)} bytes)")
        return True

    print(f"Attempting to download dataset to {output_path}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for url in URLS:
        try:
            print(f"Trying URL: {url}")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response, open(output_path, "wb") as out_file:
                total_length = response.headers.get("content-length")
                total_bytes = int(total_length) if total_length else None
                downloaded = 0
                chunk_size = 65536
                start_time = time.time()

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    out_file.write(chunk)
                    if total_bytes:
                        percent = (downloaded / total_bytes) * 100
                        speed_mb = (downloaded / (1024 * 1024)) / max(time.time() - start_time, 0.001)
                        sys.stdout.write(f"\rDownloaded {downloaded / (1024*1024):.2f} MB / {total_bytes / (1024*1024):.2f} MB ({percent:.1f}%) at {speed_mb:.2f} MB/s")
                    else:
                        sys.stdout.write(f"\rDownloaded {downloaded / (1024*1024):.2f} MB")
                    sys.stdout.flush()

            print(f"\nSuccessfully downloaded dataset to {output_path} ({os.path.getsize(output_path)} bytes)")
            return True
        except Exception as e:
            print(f"\nFailed to download from {url}: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)

    return False

if __name__ == "__main__":
    target = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", "Phishing_Email.csv")
    )
    success = download_data(target)
    if not success:
        print("Dataset download failed. Please check network connectivity or manually place Phishing_Email.csv in data/raw/")
        sys.exit(1)
