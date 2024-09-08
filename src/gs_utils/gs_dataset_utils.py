from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import math
import torch
from plyfile import PlyData
from huggingface_hub import login
import os
from huggingface_hub import login



def load_gs_dataset():
    
    dataset = load_dataset("ekolasky/gaussian-splat-hydrants")

    print(len(dataset["train"][0]["points"][0]))

    # Filter out objects with less than 2048 points
    dataset["train"] = dataset["train"].filter(lambda x: len(x["points"]) == 2048)
    dataset["test"] = dataset["test"].filter(lambda x: len(x["points"]) == 2048)

    return dataset["train"], dataset["test"]
    
class GSDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.tensor(item["points"])
    

def convert_ply_to_df(ply_file_path):

    ply = PlyData.read(ply_file_path)
    df = pd.DataFrame(ply.elements[0].data)

    df = df[['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']]
    return df

def upload_gs_dataset(examples, split_ratio=0.8):

    # Split the dataset into train and test
    train_set = Dataset.from_list(examples[:math.ceil(len(examples)*split_ratio)])
    test_set = Dataset.from_list(examples[math.ceil(len(examples)*split_ratio):])

    # Upload to Hugging Face
    print("Uploading to HF...")
    dataset_dict = DatasetDict({
        "train": train_set,
        "test": test_set
    })

    dataset_dict.push_to_hub(repo_id="ekolasky/gaussian-splat-hydrants")
    print("Finished uploading")

    