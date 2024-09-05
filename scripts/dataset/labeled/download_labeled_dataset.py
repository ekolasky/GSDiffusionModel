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
from typing import List
import shutil
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)

from src.gs_utils.convert_co3d_to_gs import add_colmap_to_sequence_folder, generate_gs_for_folder, remove_shs_from_model

def download_category_batch(category_path, link):
    # Download the batch data
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

    # Move files from potential subfolder to category_path
    category = category_path.split('/')[-1]
    for dir in os.listdir(os.path.join(category_path, category)):
        dir_path = os.path.join(category_path, category, dir)
        shutil.move(dir_path, category_path)

    # Remove zip file after extraction
    os.remove(zip_filename)

    # Remove empty category folder
    shutil.rmtree(os.path.join(category_path, category))


def process_category(category, links):
    """
    Processing each category does the following steps:
    - Get the links for the category
    - Create a folder for the category
    - Create a log file, along with function to add to log file
    - For each batch in the category:
    - Download the batch data
    - Unzip the batch data
    - Delete the zip file
    - Add COLMAP to batch data
    - Add GS to COLMAP data
    - Remove shs from models
    - Transfer models to labeled_gs/processed
    - Delete original batch data
    """

    print(f"Processing category: {category}")

    # Get links for the category
    if category not in links['full']:
        print(f"Error: Category '{category}' not found in links.json")
        return
    category_links = links['full'][category]

    # Create category folder
    category_path = os.path.join('data/labeled_gs/raw', category)
    os.makedirs(category_path, exist_ok=True)

    # Create log file
    log_file = os.path.join(category_path, 'log.txt')
    def add_to_log(message):
        with open(log_file, 'a') as f:
            f.write(message + '\\n')

    # Process each batch
    for link in tqdm(category_links, desc=f"Processing {len(category_links)} batches"):

        # Download batch
        download_category_batch(category_path, link)

        # Get frame annotations
        frame_annotations = load_dataclass_jgzip(os.path.join(category_path, "frame_annotations.jgz"), List[FrameAnnotation])

        # Get sequence names in folder, along with corresponding frame annotations
        sequence_frame_annotations = {}
        seqs_wo_pointcloud = []
        for frame_annotation in tqdm(frame_annotations, desc="Sorting frame annotations"):

            sequence_name = frame_annotation.sequence_name
            if os.path.isdir(os.path.join(category_path, sequence_name)):

                # Check that sequence folder has pointcloud.ply
                if not os.path.exists(os.path.join(category_path, sequence_name, "pointcloud.ply")):
                    if sequence_name not in seqs_wo_pointcloud:
                        add_to_log(f"{sequence_name}: Point cloud doesn't exist")
                    continue

                # Add frame annotation to dictionary
                if frame_annotation.sequence_name not in sequence_frame_annotations:
                    sequence_frame_annotations[sequence_name] = [frame_annotation]
                sequence_frame_annotations[sequence_name].append(frame_annotation)

        # Add COLMAP to sequence folders
        failed_seqs = []
        for sequence_name, frame_annotations in tqdm(sequence_frame_annotations.items(), desc="Preprocessing sequences"):
            try:
                add_colmap_to_sequence_folder(os.path.join(category_path, sequence_name), frame_annotations)
            except Exception as e:
                print(e)
                add_to_log(f"{sequence_name}: Failed during COLMAP")
                failed_seqs.append(sequence_name)
        sequence_frame_annotations = {key: value for key, value in sequence_frame_annotations.items() if key not in failed_seqs}

        # Add GS to COLMAP data
        failed_seqs = []
        for sequence_name, frame_annotations in tqdm(sequence_frame_annotations.items(), desc="Adding GS to COLMAP data"):
            try:
                generate_gs_for_folder(os.path.join(category_path, sequence_name))
            except Exception as e:
                add_to_log(f"{sequence_name}: Failed during GS")
                failed_seqs.append(sequence_name)
        sequence_frame_annotations = {key: value for key, value in sequence_frame_annotations.items() if key not in failed_seqs}

        # Remove shs from models
        failed_seqs = []
        for sequence_name, frame_annotations in tqdm(sequence_frame_annotations.items(), desc="Removing shs from models"):
            try:
                remove_shs_from_model(category_path)
                add_to_log(f"{sequence_name}: Success")
            except Exception as e:
                add_to_log(f"{sequence_name}: Failed during shs removal")
                failed_seqs.append(sequence_name)
        sequence_frame_annotations = {key: value for key, value in sequence_frame_annotations.items() if key not in failed_seqs}

        # Transfer models to labeled_gs/processed
        for sequence_name, frame_annotations in tqdm(sequence_frame_annotations.items(), desc="Moving models to processed folder"):
            # Create sequence folder in labeled_gs/processed
            sequence_path = os.path.join('data/labeled_gs/processed', category, sequence_name)
            os.makedirs(sequence_path, exist_ok=True)

            # Move models to sequence folder
            shutil.move(
                os.path.join(category_path, sequence_name, "point_cloud/iteration_5000/point_cloud.ply"),
                os.path.join(sequence_path, "pointcloud.ply")
            )

        # Delete original batch data
        for seq_name in [k for k in sequence_frame_annotations.keys()]:
            shutil.rmtree(os.path.join(category_path, seq_name))

        print(f"Finished processing category: {category}")


def main():
    parser = argparse.ArgumentParser(description="Process CO3D dataset into Gaussian Splats.")
    parser.add_argument('--category', type=str, help='Specific category to process (optional)')
    args = parser.parse_args()

    with open('data/labeled_gs/links.json', 'r') as f:
        links = json.load(f)
        available_categories = [k for k in links["full"].keys()]

    # Get the categories to process
    if args.category:
        if args.category not in available_categories:
            print(f"Error: Category '{args.category}' not found in available categories.")
            print(f"Available categories: {', '.join(available_categories)}")
            sys.exit(1)
        categories_to_process = [args.category]
    else:
        categories_to_process = available_categories

    # Check if raw path exists and if it does throw an error
    if not os.path.isdir('data/labeled_gs/raw'):
        os.makedirs('data/labeled_gs/raw', exist_ok=True)
    else:
        raise ValueError("data/labeled_gs/raw already exists")

    print(f"Categories to process: {', '.join(categories_to_process)}")

    for category in categories_to_process:
        process_category(category, links)


if __name__ == "__main__":
    main()