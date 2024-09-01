import argparse
import torch
from tqdm import tqdm
import os
from datetime import datetime
from PIL import Image
import pandas as pd
import numpy as np
from plyfile import PlyData, PlyElement
import zipfile
import shutil
import subprocess
from src.model.modeling_gst import GSTModel, GSTConfig
from src.gs_utils.training_utils import create_noise_input_vecs

def inference(model, device, render_images=True):
    
    # Draw initial noise input vectors
    x = create_noise_input_vecs(
        batch=None,
        t=model.config.timesteps,
        config=model.config
    )
    
    for t in tqdm(range(model.config.timesteps)):
        # The model predicts the denoised output directly
        with torch.no_grad():
            x = model(x)

    # Create a timestamp for the folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the output folder if it doesn't exist
    output_folder = "outputs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create a subfolder with the timestamp
    subfolder = os.path.join(output_folder, timestamp)
    os.makedirs(subfolder)

    # Save x to output folder as a .ply file
    x = x.cpu().numpy()

    # Create a DataFrame from the numpy array
    df = pd.DataFrame(x[0], columns=['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'])

    # Add f_rest_0 to f_rest_44 columns with values of 0
    for i in range(0, 45):
        df[f'f_rest_{i}'] = 0

    # Convert DataFrame to structured array
    vertex = np.array([tuple(row) for row in df.values], 
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                             ('opacity', 'f4'),
                             ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                             ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')] + [(f'f_rest_{i}', 'f4') for i in range(0, 45)])

    # Create PlyElement
    el = PlyElement.describe(vertex, 'vertex')

    # Create PlyData object and save to file
    PlyData([el]).write(os.path.join(subfolder, 'point_cloud.ply'))

    print(f"Point cloud saved as 'point_cloud.ply' in {subfolder}")

    if render_images:
        _render_images(subfolder)

def _render_images(subfolder):
    # Render the final output from different views

    # Unzip sparse.zip into the subfolder
    with zipfile.ZipFile('sparse.zip', 'r') as zip_ref:
        zip_ref.extractall(subfolder)

    # Create point_cloud/iteration_5000 directory structure
    point_cloud_dir = os.path.join(subfolder, 'point_cloud', 'iteration_5000')
    os.makedirs(point_cloud_dir, exist_ok=True)

    # Move the .ply file to the new directory
    ply_source = os.path.join(subfolder, 'point_cloud.ply')
    ply_destination = os.path.join(point_cloud_dir, 'point_cloud.ply')
    shutil.move(ply_source, ply_destination)

    print(f"Unzipped sparse.zip and moved point_cloud.ply to {point_cloud_dir}")

    # Now take images.txt from within sparse/0 and delete but the first 4, 10th, 115th, 160th, 180th lines
    # Read the images.txt file
    images_txt_path = os.path.join(subfolder, 'sparse', '0', 'images.txt')
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()

    # Keep only the specified lines
    lines_to_keep = [0, 1, 2, 3, 9, 114, 159, 179]
    filtered_lines = [lines[i] for i in lines_to_keep]

    # Write the filtered lines back to images.txt
    with open(images_txt_path, 'w') as f:
        f.writelines(filtered_lines)

    print(f"Modified images.txt in {images_txt_path}")

    # Now render using gaussian-splatting
    script_args = ["python", "../../submodules/gaussian-splatting/render.py", "-s", os.path.abspath(subfolder)]
    
    try:
        result = subprocess.run(
            script_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    
        # Check if there were any errors and print them
        if result.returncode != 0:
            print(f"Error running script. Return code: {result.returncode}")
            print("Error output:")
            print(result.stderr)
    
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = GSTConfig.load_from_json(args.model_path + "/config.json")
    model = GSTModel(config)
    model.load_state_dict(torch.load(args.model_path + "/model.pth"))
    model.to(device)
    model.eval()

    inference(model, device)
    





