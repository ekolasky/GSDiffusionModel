# GSDiffusionModel
This repository includes an unconditional diffusion model for creating 3D gaussian splats. 

## Table of contents

- [Installation](#installation)
- [How to run](#how-to-run)
- [How to train](#how-to-train)
- [How to evaluate](#how-to-evaluate)
- [How to test](#how-to-test)

## Installation


## How to run

## How to train

To train the model, you need to first download and preprocess the dataset. The downloading step downloads the CO3D dataset (which includes video frames and their corresponding camera trajectories) and the preprocessing step converts the video frames and camera trajectories into 3D gaussian splats.

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