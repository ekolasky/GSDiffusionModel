import argparse
import datasets
import pandas as pd
import json
import sys
import os
from src.gs_utils.gs_dataset_utils import convert_ply_to_df, upload_gs_dataset
from huggingface_hub import login


def main():

    parser = argparse.ArgumentParser(description="Process CO3D dataset into Gaussian Splats.")
    parser.add_argument('--category', type=str, help='Specific category to process (optional)')
    parser.add_argument('--hf_token', type=str, help='Specific category to process (optional)')
    args = parser.parse_args()

    login(token=args.hf_token)

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
        category_dir = f"data/labeled_gs/processed/{category}"
        for subdir in [f for f in os.listdir(category_dir) if os.path.isdir(category_dir + "/" + f)]:
            print(f"Converting: {subdir}")
        
            # Check if dir includes point_cloud/iteration_xxxx
            full_subdir = category_dir + "/" + subdir
            ply_file_path = full_subdir + "/pointcloud.ply"
            print(ply_file_path)
            if os.path.exists(ply_file_path):

                df = convert_ply_to_df(ply_file_path)
                examples.append({"id": subdir, "points": [row.tolist() for _, row in df.iterrows()]})

    # Convert list of examples to a datasets.Dataset
    upload_gs_dataset(examples, split_ratio=0.8)


if __name__ == "__main__":
    main()