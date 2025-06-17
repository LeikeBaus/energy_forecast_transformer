import os
import requests
import subprocess

def download_kaggle_dataset(dataset, dest_folder):
    """Download a dataset from Kaggle using the Kaggle API and unzip it."""
    os.makedirs(dest_folder, exist_ok=True)
    command = [
        'kaggle', 'datasets', 'download', '-d', dataset, '-p', dest_folder, '--unzip'
    ]
    subprocess.run(command, check=True)
    print(f"Kaggle dataset '{dataset}' downloaded and extracted to {dest_folder}.")
