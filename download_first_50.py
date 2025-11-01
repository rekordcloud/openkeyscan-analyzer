#!/usr/bin/env python3
"""
Download first 50 audio files from GiantSteps Key Dataset.
"""

import hashlib
import requests
from pathlib import Path
from tqdm import tqdm

def md5_checksum(filepath):
    """Calculate MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()

def download_file(url, output_path, expected_md5=None):
    """Download a file with progress bar and optional MD5 verification."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name, leave=False) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Verify MD5 if provided
        if expected_md5:
            actual_md5 = md5_checksum(output_path)
            if actual_md5 == expected_md5:
                return True, "MD5 OK"
            else:
                return False, f"MD5 mismatch: expected {expected_md5}, got {actual_md5}"

        return True, "Downloaded (no MD5 check)"

    except Exception as e:
        return False, str(e)

def main():
    # Paths
    dataset_dir = Path('datasets/giantsteps-key-dataset')
    audio_dir = dataset_dir / 'audio'
    md5_dir = dataset_dir / 'md5'

    # Create audio directory
    audio_dir.mkdir(exist_ok=True)

    # URLs to try
    PRIMARY_BASE_URL = "https://www.cp.jku.at/datasets/giantsteps/backup/"
    BACKUP_BASE_URL = "https://geo-samples.beatport.com/lofi/"

    # Get first 50 MD5 files (sorted)
    md5_files = sorted(md5_dir.glob('*.md5'))[:50]

    print(f"Found {len(md5_files)} files to download\n")

    success_count = 0
    error_count = 0
    errors = []

    for md5_file in tqdm(md5_files, desc="Overall progress"):
        # Get filenames
        file_id = md5_file.stem  # e.g., "1004923.LOFI"
        mp3_filename = f"{file_id}.mp3"
        audio_filepath = audio_dir / mp3_filename

        # Skip if already exists
        if audio_filepath.exists():
            # Verify MD5
            expected_md5 = md5_file.read_text().strip()
            actual_md5 = md5_checksum(audio_filepath)
            if actual_md5 == expected_md5:
                tqdm.write(f"✓ {mp3_filename} already exists (MD5 OK)")
                success_count += 1
                continue
            else:
                tqdm.write(f"⚠ {mp3_filename} exists but MD5 mismatch, re-downloading...")

        # Read expected MD5
        expected_md5 = md5_file.read_text().strip()

        # Try primary URL first
        primary_url = PRIMARY_BASE_URL + mp3_filename
        tqdm.write(f"Downloading {mp3_filename} from primary source...")
        success, message = download_file(primary_url, audio_filepath, expected_md5)

        if success and "MD5 OK" in message:
            tqdm.write(f"✓ {mp3_filename} - {message}")
            success_count += 1
            continue

        # Try backup URL
        tqdm.write(f"  Primary failed ({message}), trying backup...")
        backup_url = BACKUP_BASE_URL + mp3_filename
        success, message = download_file(backup_url, audio_filepath, expected_md5)

        if success and "MD5 OK" in message:
            tqdm.write(f"✓ {mp3_filename} - {message} (from backup)")
            success_count += 1
        else:
            tqdm.write(f"✗ {mp3_filename} - FAILED: {message}")
            error_count += 1
            errors.append((mp3_filename, message))
            # Remove failed download
            if audio_filepath.exists():
                audio_filepath.unlink()

    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    print(f"Successful: {success_count}/{len(md5_files)}")
    print(f"Failed: {error_count}/{len(md5_files)}")

    if errors:
        print("\nFailed downloads:")
        for filename, error in errors:
            print(f"  - {filename}: {error}")

    print("="*60)

if __name__ == '__main__':
    main()
