# GSDiffusionModel
This repository includes an transformer-based diffusion model for creating 3D gaussian splats. 

The repository is meant to support training on both labeled gaussian splat datasets, and unlabeled image datasets. Hopefully also eventualy video datasets. As a result, the scripts and notebooks are divided between "labeled" and "unlabeled" modes. Labeled refers to working with 3D data, and unlabeled refers to working with 2D data.

## Table of contents

- [Installation](#installation)
- [How to run](#how-to-run)
- [How to train](#how-to-train)
- [How to evaluate](#how-to-evaluate)
- [How to test](#how-to-test)

## Installation


## How to run

## How to train

To train the model, you first need to get the dataset. The dataset is stored on a S3 bucket, and can be downloaded by running following command, or by running the notebook called init_labeled_dataset.ipynb:

```bash
python scripts/dataset/download_labeled.py --category <category eg apple>
```

If you don't have access to the S3 bucket, or want to initialize the dataset from scratch, you can instead download the raw CO3D dataset and preprocess it. The notebook called create_gs_labeled_dataset.ipynb shows how to do this.





you need to first download and preprocess the dataset. The downloading step downloads the CO3D dataset (which includes video frames and their corresponding camera trajectories) and the preprocessing step converts the video frames and camera trajectories into 3D gaussian splats.

To download the CO3D dataset, run the following command:

```bash
python scripts/download_labeled_dataset.py --category <category eg apple>
```

To preprocess the dataset, run the following command:

```bash
python scripts/create_gs_labeled_dataset.py --category <category eg apple>
```

## How to evaluate

## How to test