"""
Functions to convert data from CO3D to Gaussian Splatting format.
"""

from typing import List
import os
import subprocess
import numpy as np
from plyfile import PlyData
import cv2
import sys
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)


def generate_gs_for_folder(folder_path, iteration_num, network_gui=False):
    """
    Add GS data to a folder with CO3D data and corresponding COLMAP data.
    """
    script_path = os.path.join("submodules", "gaussian-splatting", "train.py")
    print(os.path.abspath(script_path))
    print(os.path.abspath(folder_path))

    script_args = ["python", script_path, "-s", folder_path, "--model_path", os.path.abspath(folder_path)]

    if iteration_num:
        script_args += ["--iterations", iteration_num]
    
    if network_gui:
        script_args += ["--ip", "0.0.0.0", "--port", "6009"]
    else:
        script_args += ["--disable_viewer"]
    
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

def add_gs_to_colmap_folders(path, iteration_num="5_000"):
    """
    Add a GS to every colmap folder
    """

    subdirs = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    for dir in subdirs:
        # Check that dir contains COLMAP data
        if all([f != "sparse" for f in os.listdir(os.path.join(path,dir))]):
            print(f"Missing colmap data: {dir}")
            continue

        # Check that there isn't already a gs in folder
        if os.path.isdir(os.path.join(path, dir, "point_cloud", f"iteration_{iteration_num}")):
            print(f"Already a GS in {dir}")
            continue

        # Generate gaussian splat
        print(f"Generating GS for {dir}")
        generate_gs_for_folder(os.path.join(path, dir), iteration_num=iteration_num, network_gui=False)
            

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

        cameras_list = []
        images_list = []

        if not os.path.isdir(os.path.join(path, sequence_name)):
            print(f"Could not find {sequence_name}")
            continue

        if not os.path.exists(os.path.join(path, sequence_name, "pointcloud.ply")):
            print(f"Point cloud doesn't exist in {sequence_name}")
            continue

        for frame_annotation in frame_annotations:

            # Get camera data
            width = frame_annotation.image.size[1]
            height = frame_annotation.image.size[0]
            s = (min(height, width) - 1) / 2
            # fx = 2 * (width - 1) / frame_annotation.viewpoint.focal_length[1]
            fy = (width - 1) * frame_annotation.viewpoint.focal_length[0] / 2
            px = -frame_annotation.viewpoint.principal_point[0] * s + (width - 1) / 2
            py = -frame_annotation.viewpoint.principal_point[1] * s + (height - 1) / 2
            camera_data = {
                "camera_id": len(cameras_list) + 1,
                "model": "PINHOLE",
                "width": width,
                "height": height,
                "focal_length_x": fy,
                "focal_length_y": fy,
                "principal_point": (px, py)
            }

            # Check if camera_data already exists in cameras_list
            duplicate_index = None
            for i, existing_camera in enumerate(cameras_list):
                if (existing_camera['model'] == camera_data['model'] and
                    existing_camera['width'] == camera_data['width'] and
                    existing_camera['height'] == camera_data['height'] and
                    existing_camera['focal_length_x'] == camera_data['focal_length_x'] and
                    existing_camera['focal_length_y'] == camera_data['focal_length_y'] and
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
            T = np.array(frame_annotation.viewpoint.T)
            T[:2] *= -1
            image_data = {
                "image_id": frame_annotation.frame_number,
                "qw": image_quaternion[0],
                "qx": image_quaternion[1],
                "qy": image_quaternion[2],
                "qz": image_quaternion[3],
                "tx": T[0],
                "ty": T[1],
                "tz": T[2],
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

        remove_background_from_images(path, sequence_name)

        print(f"Removed background from images in {sequence_name}")



def remove_background_from_images(path: str, sequence_name: str):
    images_path = os.path.join(path, sequence_name, "images")
    masks_path = os.path.join(path, sequence_name, "masks")

    for image_name in os.listdir(images_path):
        if image_name.endswith(".jpg"):
            image_path = os.path.join(images_path, image_name)
            mask_name = image_name.replace(".jpg", ".png")
            mask_path = os.path.join(masks_path, mask_name)

            # Read the image and mask
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                raise Exception(f"Could not read image or mask for {image_name}")
                continue

            # Create a 3-channel mask
            mask_3_channel = cv2.merge([mask, mask, mask])

            # Remove the background
            image_no_bg = cv2.bitwise_and(image, mask_3_channel)

            # Save the image back to the images folder
            cv2.imwrite(image_path, image_no_bg)



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
            f"{camera_data['focal_length_x']} " +
            f"{camera_data['focal_length_y']} " +
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

def _adjust_pitch_by_degrees(q, pitch_degrees):
    pitch_radians = np.deg2rad(pitch_degrees)
    q_pitch = np.array([np.cos(pitch_radians / 2), np.sin(pitch_radians / 2), 0, 0])
    
    # Quaternion multiplication (q_pitch * q)
    w1, x1, y1, z1 = q_pitch
    w2, x2, y2, z2 = q
    
    q_new = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])
    
    return q_new
    

def _convert_rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion.
    
    Args:
        rotation (numpy.ndarray): 3x3 rotation matrix
    
    Returns:
        numpy.ndarray: Quaternion in the form [w, x, y, z]
    """

    R[:, :2] *= -1
    
    
    # Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    Rxx, Rxy, Rxz, Ryx, Ryy, Ryz, Rzx, Rzy, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    
    return qvec
    
    # trace = np.trace(rotation)
    # if trace > 0:
    #     S = np.sqrt(trace + 1.0) * 2
    #     w = 0.25 * S
    #     x = (rotation[2, 1] - rotation[1, 2]) / S
    #     y = (rotation[0, 2] - rotation[2, 0]) / S
    #     z = (rotation[1, 0] - rotation[0, 1]) / S
    # elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
    #     S = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
    #     w = (rotation[2, 1] - rotation[1, 2]) / S
    #     x = 0.25 * S
    #     y = (rotation[0, 1] + rotation[1, 0]) / S
    #     z = (rotation[0, 2] + rotation[2, 0]) / S
    # elif rotation[1, 1] > rotation[2, 2]:
    #     S = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
    #     w = (rotation[0, 2] - rotation[2, 0]) / S
    #     x = (rotation[0, 1] + rotation[1, 0]) / S
    #     y = 0.25 * S
    #     z = (rotation[1, 2] + rotation[2, 1]) / S
    # else:
    #     S = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
    #     w = (rotation[1, 0] - rotation[0, 1]) / S
    #     x = (rotation[0, 2] + rotation[2, 0]) / S
    #     y = (rotation[1, 2] + rotation[2, 1]) / S
    #     z = 0.25 * S
    
    # return np.array([w, x, y, z])

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

