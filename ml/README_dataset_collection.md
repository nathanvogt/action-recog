# Dataset Collection and Training Workflow

This document explains how to use the separate data collection and training workflow to avoid re-collecting expensive pose sequence data.

## Overview

The workflow is split into two main steps:

1. **Data Collection**: Sample and process pose sequences, then save to disk
2. **Training**: Load pre-saved data and train the model

This separation allows you to:

- Collect data once and reuse it for multiple training experiments
- Experiment with different model architectures without re-collecting data
- Share pre-processed datasets between experiments

## Step 1: Data Collection

Use `collect_dataset.py` to sample pose sequences and save them:

```bash
# Using configuration file
python collect_dataset.py --config configs/data_collection.yaml

# Or with command line arguments
python collect_dataset.py \
    --subject S03 \
    --exercise dumbbell_biceps_curls \
    --n-samples 5000 \
    --save-path ./datasets/S03_dumbbell_biceps_curls
```

### Data Collection Configuration

Create a YAML file (e.g., `configs/data_collection.yaml`):

```yaml
subject: "S03"
exercise: "dumbbell_biceps_curls"
dataset-root: "train"
c: 24
m: 4
tol: 20
n-samples: 5000 # More samples for better dataset
save-path: "./datasets/S03_dumbbell_biceps_curls"
```

### What Gets Saved

The script saves three files in the specified directory:

- `features.npy`: Processed pose sequence features (numpy array)
- `labels.npy`: Binary labels (0=negative, 1=positive rep)
- `metadata.pkl`: Dataset metadata (parameters, statistics, etc.)

## Step 2: Training

Use `train_supervised.py` with the `--load-dataset` option:

```bash
# Using configuration file
python train_supervised.py --config configs/supervised_training_preloaded.yaml

# Or with command line arguments
python train_supervised.py \
    --load-dataset ./datasets/S03_dumbbell_biceps_curls \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 0.001
```

### Training Configuration

Create a YAML file (e.g., `configs/supervised_training_preloaded.yaml`):

```yaml
# Training with pre-saved dataset
load-dataset: "./datasets/S03_dumbbell_biceps_curls"

# Training parameters
epochs: 20
batch-size: 32
learning-rate: 0.001

# Model parameters
hidden-dim: 64
n-layers: 3
save-path: "./models/supervised_preloaded"
```

## Testing Dataset Loading

You can test that a dataset loads correctly:

```bash
python collect_dataset.py --load-test ./datasets/S03_dumbbell_biceps_curls
```

## Example Complete Workflow

```bash
# 1. Collect dataset (this is the slow step)
python collect_dataset.py --config configs/data_collection.yaml

# 2. Train model (fast, can be repeated with different parameters)
python train_supervised.py --config configs/supervised_training_preloaded.yaml

# 3. Experiment with different architectures
python train_supervised.py \
    --load-dataset ./datasets/S03_dumbbell_biceps_curls \
    --hidden-dim 128 \
    --n-layers 4 \
    --epochs 30 \
    --save-path ./models/experiment_1

# 4. Try different hyperparameters
python train_supervised.py \
    --load-dataset ./datasets/S03_dumbbell_biceps_curls \
    --learning-rate 0.0001 \
    --batch-size 64 \
    --epochs 50 \
    --save-path ./models/experiment_2
```

## Benefits

- **Time Saving**: Data collection happens once, not every training run
- **Reproducibility**: Same dataset for all experiments
- **Experimentation**: Easy to try different model architectures and hyperparameters
- **Sharing**: Pre-processed datasets can be shared between researchers

## Legacy Support

The original `train_supervised.py` still works for fresh data collection:

```bash
python train_supervised.py --config configs/supervised_training.yaml
```

This will collect data fresh each time (slower but doesn't require pre-collection).
