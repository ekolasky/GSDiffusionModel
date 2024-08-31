"""
Functions to convert data from CO3D to Gaussian Splatting format.
"""

from typing import List
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)

def convert_co3d_folder_to_gs(img_path):
    """
    Add GS data to a folder with CO3D data.
    """
    pass

def add_colmap_to_category_folders(path):
    """
    Add COLMAP data to a folder with CO3D data.
    """

    # Get frame annotations
    frame_annotations = load_dataclass_jgzip(path, List[FrameAnnotation])

    # Iterate through each folder
    for frame_annotation in frame_annotations:
        # Get the image path
        img_path = os.path.join(path, frame_annotation.sequence_name)

        # Get camera data
        camera_data = {}
        camera_data["camera_id"] = None
        camera_data["model"] = None
        camera_data["width"] = None
        camera_data["height"] = None
        camera_data["focal_length"] = None
        camera_data["principal_point"] = None
        camera_data["distortion_params"] = None

        # Create camera.txt




