import argparse
import datasets
import pandas as pd
import json
import sys
import os
from src.gs_utils.gs_dataset_utils import convert_ply_to_df, upload_gs_dataset

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

    examples = []
    for category in categories_to_process:
        category_dir = f"data/labeled_gs/raw/{category}"
        for batchdir in [f for f in os.listdir(category_dir) if os.path.isdir(category_dir + "/" + f)]:
            full_batchdir = category_dir + "/" + batchdir
            for subdir in [f for f in os.listdir(full_batchdir) if os.path.isdir(full_batchdir + "/" + f)]:
            
                # Check if dir includes point_cloud/iteration_xxxx
                full_subdir = category_dir + "/" + batchdir + "/" + subdir
                print(subdir)
                if os.path.exists(f"{full_subdir}/point_cloud") and \
                    any([f for f in os.listdir(f"{full_subdir}/point_cloud") if f.startswith("iteration_")]):
                    # Get ply file from largest iteration
                    iteration_dirs = [f for f in os.listdir(f"{full_subdir}/point_cloud") if f.startswith("iteration_")]
                    iteration_dirs.sort(key=lambda x: int(x.split("_")[1]))
                    ply_file_path = f"{full_subdir}/point_cloud/{iteration_dirs[-1]}/point_cloud.ply"
    
                    df = convert_ply_to_df(ply_file_path)
                    examples.append(df)

    # Convert list of examples to a datasets.Dataset
    print(len(examples))
    upload_gs_dataset(examples, split_ratio=0.8)


if __name__ == "__main__":
    main()