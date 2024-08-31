"""
Functions to convert data from CO3D to Gaussian Splatting format.
"""

from typing import List
import os
import subprocess
import numpy as np
from plyfile import PlyData
import sys
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)


def generate_gs_for_folder(folder_path):
    """
    Add GS data to a folder with CO3D data and corresponding COLMAP data.
    """
    script_path = os.path.join("submodules", "gaussian-splatting", "train.py")
    print(os.path.abspath(script_path))
    print(os.path.abspath(folder_path))
    
    try:
        process = subprocess.Popen(["python", script_path, "-s", folder_path], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True, 
                                   bufsize=1, 
                                   universal_newlines=True)
        
        # Function to handle output
        def print_output(stream):
            for line in stream:
                print(line, end='')
                sys.stdout.flush()
        
        # Print output in real-time
        while process.poll() is None:
            print_output(process.stdout)
            print_output(process.stderr)
        
        # Print any remaining output
        print_output(process.stdout)
        print_output(process.stderr)
        
        if process.returncode != 0:
            print(f"Error running script. Return code: {process.returncode}")
    except Exception as e:
        print(f"An error occurred: {e}")
    

def add_colmap_to_category_folders(path):
    """
    Add COLMAP data to a folder with CO3D data.
    """

    # Get frame annotations
    frame_annotations = load_dataclass_jgzip(os.path.join(path, "frame_annotations.jgz"), List[FrameAnnotation])

    cameras_list = []

    # Seperate frame annotations by sequence name
    sequence_frame_annotations = {}
    for frame_annotation in frame_annotations:
        if frame_annotation.sequence_name not in sequence_frame_annotations:
            sequence_frame_annotations[frame_annotation.sequence_name] = []
        sequence_frame_annotations[frame_annotation.sequence_name].append(frame_annotation)

    # Iterate through each folder
    for sequence_name, frame_annotations in sequence_frame_annotations.items():

        if sequence_name != "106_12653_23216":
            continue

        cameras_list = []
        images_list = []

        for frame_annotation in frame_annotations:

            # Get camera data
            camera_data = {
                "camera_id": len(cameras_list) + 1,
                "model": "PINHOLE",
                "width": frame_annotation.image.size[0],
                "height": frame_annotation.image.size[1],
                "focal_length": frame_annotation.viewpoint.focal_length[0],
                "principal_point": frame_annotation.viewpoint.principal_point
            }

            if frame_annotation.viewpoint.focal_length[0] != frame_annotation.viewpoint.focal_length[1]:
                raise ValueError("Focal length is not the same for x and y.")

            # Check if camera_data already exists in cameras_list
            duplicate_index = None
            for i, existing_camera in enumerate(cameras_list):
                if (existing_camera['model'] == camera_data['model'] and
                    existing_camera['width'] == camera_data['width'] and
                    existing_camera['height'] == camera_data['height'] and
                    existing_camera['focal_length'] == camera_data['focal_length'] and
                    existing_camera['principal_point'] == camera_data['principal_point']):
                    duplicate_index = i
                    break
            
            if duplicate_index is None:
                # No duplicate found, add the new camera_data
                cameras_list.append(camera_data)
            else:
                # Duplicate found, update camera_data's camera_id
                camera_data['camera_id'] = cameras_list[duplicate_index]['camera_id']
            
            image_quaternion = _convert_rotation_to_quaternion(np.array(frame_annotation.viewpoint.R))
            image_data = {
                "image_id": frame_annotation.frame_number,
                "qw": image_quaternion[0],
                "qx": image_quaternion[1],
                "qy": image_quaternion[2],
                "qz": image_quaternion[3],
                "tx": frame_annotation.viewpoint.T[0],
                "ty": frame_annotation.viewpoint.T[1],
                "tz": frame_annotation.viewpoint.T[2],
                "camera_id": camera_data['camera_id'],
                "image_name": frame_annotation.image.path.split('/')[-1]
            }
            images_list.append(image_data)


        image_txt = _get_image_txt(images_list)
        camera_txt = _get_camera_txt(cameras_list)
        points3D_txt = _convert_ply_to_points3D(os.path.join(path, sequence_name, "pointcloud.ply"))

        
        metadata_path = os.path.join(path, sequence_name, "sparse", "0")
        os.makedirs(metadata_path, exist_ok=True)

        with open(os.path.join(metadata_path, "cameras.txt"), "w") as f:
            f.write(camera_txt)

        with open(os.path.join(metadata_path, "images.txt"), "w") as f:
            f.write(image_txt)

        with open(os.path.join(metadata_path, "points3D.txt"), "w") as f:
            f.write(points3D_txt)

        print(f"Added COLMAP data to {sequence_name}")

        break
    

def _get_camera_txt(cameras_list: List[dict]):
    """
    Get the camera.txt file from a frame annotation.
    """
    
    # Create camera.bin
    camera_txt = f"# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[6]\n# Number of cameras: {len(cameras_list)}\n"

    for camera_data in cameras_list:        
        camera_txt += (
            f"{camera_data['camera_id']} " +
            f"{camera_data['model']} " +
            f"{camera_data['width']} " +
            f"{camera_data['height']} " +
            f"{camera_data['focal_length']} " +
            f"{camera_data['principal_point'][0]} " +
            f"{camera_data['principal_point'][1]}\n"
        )
        
    # Remove the last newline character
    camera_txt = camera_txt[:-1]

    return camera_txt


def _get_image_txt(images_list: List[dict]):
    """
    Get the image.txt file from a frame annotation.
    """

    image_txt = "# Image list with two lines of data per image:\n"
    image_txt += "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
    image_txt += "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
    image_txt += f"# Number of images: {len(images_list)}, mean observations per image: {2500}\n"

    for image_data in images_list:
        image_txt += (
            f"{image_data['image_id']} " +
            f"{image_data['qw']} " +
            f"{image_data['qx']} " +
            f"{image_data['qy']} " +
            f"{image_data['qz']} " +
            f"{image_data['tx']} " +
            f"{image_data['ty']} " +
            f"{image_data['tz']} " +
            f"{image_data['camera_id']} " +
            f"{image_data['image_name']}\n\n"
        )

    image_txt = image_txt[:-1]
    
    return image_txt
    

def _convert_rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion.
    
    Args:
        rotation (numpy.ndarray): 3x3 rotation matrix
    
    Returns:
        numpy.ndarray: Quaternion in the form [w, x, y, z]
    """
    
    trace = np.trace(rotation)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (rotation[2, 1] - rotation[1, 2]) / S
        y = (rotation[0, 2] - rotation[2, 0]) / S
        z = (rotation[1, 0] - rotation[0, 1]) / S
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        S = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
        w = (rotation[2, 1] - rotation[1, 2]) / S
        x = 0.25 * S
        y = (rotation[0, 1] + rotation[1, 0]) / S
        z = (rotation[0, 2] + rotation[2, 0]) / S
    elif rotation[1, 1] > rotation[2, 2]:
        S = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
        w = (rotation[0, 2] - rotation[2, 0]) / S
        x = (rotation[0, 1] + rotation[1, 0]) / S
        y = 0.25 * S
        z = (rotation[1, 2] + rotation[2, 1]) / S
    else:
        S = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
        w = (rotation[1, 0] - rotation[0, 1]) / S
        x = (rotation[0, 2] + rotation[2, 0]) / S
        y = (rotation[1, 2] + rotation[2, 1]) / S
        z = 0.25 * S
    
    return np.array([w, x, y, z])

def _convert_ply_to_points3D(ply_file):
    # Read the .ply file
    ply_data = PlyData.read(ply_file)
    
    # Extract the vertices (3D points) and possibly colors
    vertices = ply_data['vertex']
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']
    if 'red' in vertices and 'green' in vertices and 'blue' in vertices:
        r = vertices['red']
        g = vertices['green']
        b = vertices['blue']
    else:
        r = g = b = [255] * len(x)  # Default color if not available
    
    # Open the output file
    points3D_txt = "# 3D point list with one line of data per point:\n"
    points3D_txt += "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
    points3D_txt += f"# Number of points: {len(x)}, mean track length: 0.0\n"
    
    for i in range(len(x)):
        # Assign POINT3D_ID and ERROR (can be set to a default value like 0.0)
        point_id = i + 1
        error = 0.0
        # Prepare the line for points3D.txt
        line = f"{point_id} {x[i]} {y[i]} {z[i]} {r[i]} {g[i]} {b[i]} {error}"
        # If you have TRACK data, you can append it here. Otherwise, leave it empty.
        points3D_txt += line + "\n"

    return points3D_txt