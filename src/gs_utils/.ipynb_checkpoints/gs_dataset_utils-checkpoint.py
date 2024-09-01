from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import math
from plyfile import PlyData
from huggingface_hub import login
from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))


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

    df = df[['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']]
    return df

def upload_gs_dataset(examples, split_ratio=0.8):

    # Format examples for uploading
    formatted_examples = []
    for i, example in enumerate(examples):
        formatted_examples.append({"idx": i, "points": [row.tolist() for _, row in example.iterrows()]})

    # Load existing dataset
    # train_dataset, test_dataset = load_gs_dataset()

    # Merge train and test datasets, add new examples, and split
    # df = pd.concat([train_dataset, test_dataset])

    # Split the dataset into train and test
    train_set = Dataset.from_list(formatted_examples[:math.ceil(len(formatted_examples)*split_ratio)])
    test_set = Dataset.from_list(formatted_examples[math.ceil(len(formatted_examples)*split_ratio):])

    # Replace csv files with new datasets
    # train_dataset.to_csv("data/labeled_gs/train.csv", index=False)
    # test_dataset.to_csv("data/labeled_gs/test.csv", index=False)

    # Upload to Hugging Face
    print("Uploading to HF...")
    dataset_dict = DatasetDict({
        "train": train_set,
        "test": test_set
    })

    dataset_dict.push_to_hub(repo_id="ekolasky/gaussian-splat-hydrants")
    print("Finished uploading")

    