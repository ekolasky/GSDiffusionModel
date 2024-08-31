

"""
This script downloads the CO3D dataset. It saves each category of the dataset into a seperate folder.
"""

import os
import argparse
import sys
import json
import requests
import zipfile
from tqdm import tqdm
import shutil

def process_category(category, links):
    print(f"Processing category: {category}")
    
    if category not in links['full']:
        print(f"Error: Category '{category}' not found in links.json")
        return
    
    category_links = links['full'][category]
    
    # Create category folder
    category_path = os.path.join('data/labeled_gs/raw', category)
    os.makedirs(category_path, exist_ok=True)
    
    # Download and unzip files
    for link in tqdm(category_links, desc=f"Downloading {category} data"):
        zip_filename = os.path.join(category_path, os.path.basename(link))
        
        # Download zip file
        response = requests.get(link, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_filename, 'wb') as file, tqdm(
            desc=zip_filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
        
        # Unzip file with progress bar
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            total_files = len(zip_ref.infolist())  # Total number of files to extract
            for file in tqdm(zip_ref.infolist(), total=total_files, desc="Extracting"):
                zip_ref.extract(file, category_path)
        
        # Remove zip file after extraction
        os.remove(zip_filename)
    
    print(f"Finished processing category: {category}")


def main():
    parser = argparse.ArgumentParser(description="Process CO3D dataset into Gaussian Splats.")
    parser.add_argument('--category', type=str, help='Specific category to process (optional)')
    args = parser.parse_args()

    with open('data/labeled_gs/links.json', 'r') as f:
        links = json.load(f)
        available_categories = [k for k in links["full"].keys()]

    if args.category:
        if args.category not in available_categories:
            print(f"Error: Category '{args.category}' not found in available categories.")
            print(f"Available categories: {', '.join(available_categories)}")
            sys.exit(1)
        categories_to_process = [args.category]
    else:
        categories_to_process = available_categories

    # Check if raw path exists and if it does delete any folders that we're replacing
    if not os.path.isdir('data/labeled_gs/raw'):
        os.makedirs('data/labeled_gs/raw', exist_ok=True)
    else:
        for category in categories_to_process:
            category_path = os.path.join('data/labeled_gs/raw', category)
            if os.path.isdir(category_path):
                print(f"Deleting: ${category_path}")
                shutil.rmtree(category_path)
                

    print(f"Categories to process: {', '.join(categories_to_process)}")
    
    for category in categories_to_process:
        process_category(category, links)


if __name__ == "__main__":
    main()
