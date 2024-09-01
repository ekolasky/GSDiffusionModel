from datasets import load_dataset, upload_dataset
import pandas as pd
from plyfile import PlyData

def download_gs_dataset():
    dataset = load_dataset("rotten_tomatoes")

    # Save train and test to separate directories
    dataset['train'].to_csv("data/labeled_gs/train.csv", index=False)
    dataset['test'].to_csv("data/labeled_gs/test.csv", index=False)

    return dataset


def load_gs_dataset():
    
    # Try loading from data directory
    try:
        train_dataset = load_dataset("csv", data_files="data/labeled_gs/train.csv")
        test_dataset = load_dataset("csv", data_files="data/labeled_gs/test.csv")
        return train_dataset, test_dataset
    except FileNotFoundError:
        # If not found, download the dataset
        return download_gs_dataset()

def convert_ply_to_df(ply_file_path):
    ply = PlyData.read(ply_file_path)
    df = pd.DataFrame(ply.elements[0].data)

    df = df[['x', 'y', 'z', 'nx', 'ny', 'nz']]
    return df

def upload_gs_dataset(examples, split_ratio=0.8):

    # Create a dataframe
    df = pd.DataFrame(examples)

    # Load existing dataset
    train_dataset, test_dataset = load_gs_dataset()

    # Merge train and test datasets, add new examples, and split
    train_dataset = pd.concat([train_dataset, df])
    test_dataset = pd.concat([test_dataset, df])

    # Split the dataset into train and test
    train_dataset = train_dataset.sample(frac=split_ratio, random_state=42)
    test_dataset = test_dataset.drop(train_dataset.index)

    # Replace csv files with new datasets
    train_dataset.to_csv("data/labeled_gs/train.csv", index=False)
    test_dataset.to_csv("data/labeled_gs/test.csv", index=False)

    # Upload to Hugging Face
    upload_dataset(train_dataset, "data/labeled_gs/train.csv")
    upload_dataset(test_dataset, "data/labeled_gs/test.csv")