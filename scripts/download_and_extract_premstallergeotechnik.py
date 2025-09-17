#!/usr/bin/env python3
"""
Script to download the PremstallerGeotechnik CPT database zip file and extract it to data/raw.
"""
import os
import requests
import zipfile
from io import BytesIO

def main():
    url = "https://www.tugraz.at/fileadmin/user_upload/Institute/IBG/Datenbank/Database_CPT_PremstallerGeotechnik.zip"
    extract_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Downloading {url} ...")
    response = requests.get(url)
    response.raise_for_status()
    print("Download complete. Extracting...")
    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        zf.extractall(extract_dir)
    print(f"Extraction complete. Files extracted to {extract_dir}")

if __name__ == "__main__":
    main()
