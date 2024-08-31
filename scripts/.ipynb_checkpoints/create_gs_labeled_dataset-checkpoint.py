
"""
This script processes the CO3D dataset into Gaussian Splats. By default it runs through every category folder in data/labeled_gs/raw.
If you provide a category as an argument, it will only process that category.
"""

import os
import argparse
import json
import sys
from src.gs_utils.convert_co3d_to_gs import add_colmap_to_category_folders


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



def main():
    parser = argparse.ArgumentParser(description="Process CO3D dataset into Gaussian Splats.")
    parser.add_argument('--category', type=str, help='Specific category to process (optional)')
    args = parser.parse_args()

    with open('data/labeled_gs/links.json', 'r') as f:
        links = json.load(f)
        available_categories = [k for k in links["full"].keys()]

    # Double check that the category is valid
    if args.category:
        if args.category not in available_categories:
            print(f"Error: Category '{args.category}' not found in available categories.")
            print(f"Available categories: {', '.join(available_categories)}")
            sys.exit(1)
        categories_to_process = [args.category]
    else:
        categories_to_process = available_categories

    subfolders = check_directory_structure()

    # Check that all subfolders exist for all categories
    if any([category not in subfolders for category in categories_to_process]):
        raise Error("Subfolder missing for some categories")

    for category in categories_to_process:
        print(f"Processing category: {category}")
        category_path = os.path.join('data/labeled_gs/raw', category)
        for folder in [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]:
            print(f"Processing folder: {os.path.join(category_path, folder)}")
            add_colmap_to_category_folders(os.path.join(category_path, folder))
        
            

if __name__ == "__main__":
    main()
