
"""
This script processes the CO3D dataset into Gaussian Splats. By default it runs through every category folder in data/labeled_gs/raw.
If you provide a category as an argument, it will only process that category.
"""

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