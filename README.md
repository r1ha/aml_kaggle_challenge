# AML Kaggle Competition - Text-to-Image Embedding Alignment

*This README was written with AI assistance.*

## Overview

This project implements a neural network pipeline for aligning text embeddings with image embeddings, designed for a Kaggle competition. The goal is to train a model that can map textual descriptions to their corresponding visual representations in embedding space.

## Data Structure

The project expects the following data structure:

```
data/
├── train/
│   ├── train.npz              # Training data with text and image embeddings
│   ├── captions.txt           # Training captions
│   └── Images/                # Training images folder
└── test/
    ├── test.clean.npz         # Test data with text embeddings
    └── captions.txt           # Test captions
```

### Data Files Description:
- **train.npz**: Contains `captions/embeddings` (1024-dim text), `images/embeddings` (1536-dim image), `captions/text`, `images/names`
- **test.clean.npz**: Contains `captions/ids`, `captions/text`, `captions/embeddings` for inference

## Model Architecture

The pipeline uses a **2-layer MLP** with maximum semantic expansion:

```
Text Embedding (1024) → Hidden Layer (4096) → Image Embedding (1536)
```

**Key Features:**
- **Massive expansion**: 4096 hidden units to capture rich text semantics
- **Cosine Similarity Loss**: Optimizes for directional alignment between embeddings
- **Dropout regularization**: 0.2 dropout rate to prevent overfitting
- **GPU optimization**: Full GPU utilization for training and inference

## Pipeline Usage

### Main Pipeline (`pipeline.py`)
The main script that handles both training and inference:

```bash
python pipeline.py
```

**What it does:**
1. **Training Phase**: 
   - Loads training data directly to GPU
   - Trains the MLP for 20 epochs with cosine similarity loss
   - Saves model checkpoints every 5 epochs
   - Saves final model as `pipeline_final_model.pth`

2. **Testing Phase**:
   - Loads test data and generates predictions
   - Performs batch inference with GPU optimization
   - Evaluates model performance on training subset
   - Creates submission file `pipeline_submission.csv`

### Verification Script (`verif.py`)
Detailed analysis and visualization tool:

```bash
python verif.py
```

**Features:**
- Distance analysis (Euclidean, Manhattan, Cosine)
- Performance comparison vs random baselines
- Visualization of results with sample images
- Statistical metrics and improvement percentages

## Model Files

Generated model files:
- `pipeline_final_model.pth`: Final trained model weights
- `pipeline_model_checkpoint_epoch_X.pth`: Training checkpoints
- `pipeline_submission.csv`: Kaggle submission file

## Project Structure

```
├── src/
│   ├── dataset_loader.py      # Data loading utilities
│   ├── pipeline.py           # Main training and inference pipeline
│   ├── test.py               # Legacy test script
│   └── verif.py              # Model verification and analysis
├── playground/               # Experimental scripts
├── models_archive/           # Saved model files
├── submissions_archive/      # Previous submissions
├── data/                     # Dataset (structure described above)
└── requirements.txt          # Python dependencies
```

## Technical Details

- **Input Dimension**: 1024 (text embeddings)
- **Hidden Dimension**: 4096 (semantic expansion)
- **Output Dimension**: 1536 (image embeddings)
- **Loss Function**: Cosine Similarity Loss (1 - cosine_similarity)
- **Optimizer**: Adam (lr=0.001)
- **Regularization**: Dropout (0.2)
- **Training Epochs**: 20
- **Batch Size**: 64 (training), 256 (inference)

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Main dependencies:
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- PIL (Pillow)
- Matplotlib

## Usage Instructions

1. **Setup data**: Place your dataset in the `data/` folder following the structure above
2. **Train model**: Run `python pipeline.py` to train and generate predictions
3. **Verify results**: Run `python verif.py` to analyze model performance
4. **Submit**: Use `pipeline_submission.csv` for Kaggle submission

The pipeline is optimized for GPU usage and will automatically detect and use CUDA if available.