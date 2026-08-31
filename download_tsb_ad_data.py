import ssl
import urllib.request
import zipfile
import os
import sys

# Bypass SSL
ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://www.thedatum.org/datasets/TSB-AD-U.zip'
out_dir = 'TSB-AD/datasets'
os.makedirs(out_dir, exist_ok=True)
zip_path = os.path.join(out_dir, 'TSB-AD-U.zip')

print(f'Downloading TSB-AD-U.zip (this may take a while)...')

# Add a User-Agent header to avoid 403 Forbidden
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})

with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
    length = int(response.getheader('content-length', 0))
    downloaded = 0
    block_size = 1024 * 1024 # 1MB chunks
    while True:
        buffer = response.read(block_size)
        if not buffer:
            break
        downloaded += len(buffer)
        out_file.write(buffer)
        if length:
            percent = downloaded * 100 / length
            sys.stdout.write(f"\rDownloading: {percent:.1f}%")
            sys.stdout.flush()

print('\nExtracting files...')

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(out_dir)

print('Cleaning up zip file...')
os.remove(zip_path)

print('Done! Univariate datasets downloaded successfully.')
