#!/usr/bin/env python3
"""
Download Simple Wikipedia dump for benchmarking.
This script downloads the latest simple Wikipedia dump and saves it to the data folder.
"""

import urllib.request
import os
import sys
from pathlib import Path


def get_latest_dump_url():
    """
    Find the latest simple Wikipedia dump URL.
    Returns the URL to the latest bzip2 compressed XML dump.
    """
    base_url = "https://dumps.wikimedia.org/other/mediawiki_content_current/simplewiki/"

    # For now, use the known latest dump
    # In production, you could scrape the directory to find the latest
    dump_url = "https://dumps.wikimedia.org/other/mediawiki_content_current/simplewiki/2026-02-01/xml/bzip2/simplewiki-2026-02-01-p1p1251695.xml.bz2"

    return dump_url


def download_with_progress(url, dest_path):
    """Download file with progress bar."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {dest_path}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100.0 / total_size, 100) if total_size > 0 else 0
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(
            f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)"
        )
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False


def main():
    # Ensure data directory exists
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Get the dump URL
    dump_url = get_latest_dump_url()
    filename = dump_url.split("/")[-1]
    dest_path = data_dir / filename

    # Check if file already exists
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        response = input(f"File already exists ({size_mb:.1f} MB). Re-download? (y/N): ")
        if response.lower() != 'y':
            print("Using existing file.")
            return

    # Download the dump
    success = download_with_progress(dump_url, dest_path)

    if success:
        # Show file size
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"Downloaded file size: {size_mb:.1f} MB")
        print(f"\nYou can now run: python parse_wikipedia.py {dest_path}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
