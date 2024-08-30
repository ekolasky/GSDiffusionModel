

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

def check_directory_structure():
    base_path = 'data/labeled_gs'
    raw_path = os.path.join(base_path, 'raw')

    if not os.path.isdir(raw_path):
        raise FileNotFoundError(f"The '/raw' subfolder does not exist in {base_path}.")

    subfolders = [f for f in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, f))]
    
    if not subfolders:
        raise FileNotFoundError(f"No category folders found within {raw_path}.")

    print(f"Directory structure verified. Found {len(subfolders)} category folders.")
    return subfolders

def process_category(category):
    print(f"Processing category: {category}")
    
    # Load links from JSON file
    with open('data/labeled_gs/links.json', 'r') as f:
        links_data = json.load(f)
    
    if category not in links_data['full']:
        print(f"Error: Category '{category}' not found in links.json")
        return
    
    category_links = links_data['full'][category]
    
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
        
        # Unzip file
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(category_path)
        
        # Remove zip file after extraction
        os.remove(zip_filename)
    
    print(f"Finished processing category: {category}")


def main():
    parser = argparse.ArgumentParser(description="Process CO3D dataset into Gaussian Splats.")
    parser.add_argument('--category', type=str, help='Specific category to process (optional)')
    args = parser.parse_args()

    try:
        available_categories = check_directory_structure()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.category:
        if args.category not in available_categories:
            print(f"Error: Category '{args.category}' not found in available categories.")
            print(f"Available categories: {', '.join(available_categories)}")
            sys.exit(1)
        categories_to_process = [args.category]
    else:
        categories_to_process = available_categories

    print(f"Categories to process: {', '.join(categories_to_process)}")
    
    for category in categories_to_process:
        process_category(category)


if __name__ == "__main__":
    main()
