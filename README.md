# AML Kaggle Competition - Text-to-Image Embedding Alignment

*This README was written with AI assistance and updated by Team FrenchML.*

## 🚀 Overview

This project implements a neural network pipeline for the Advanced Machine Learning Kaggle competition. The goal is to perform **latent space translation**, mapping 1024-dimension text embeddings to their corresponding 1536-dimension image embeddings in a shared latent space.

The project explores the evolution from a simple Multi Layer Perceptron (MLP) with cosine similarity loss to a more advanced **Residual Network** trained with a contrastive **InfoNCE loss**, which yielded our best results.

## 💾 Data Structure

The project data expects the following structure:

```
data/
├── train/
│   ├── train.npz           # Training data with text and image embeddings
│   ├── captions.txt        # Training captions
│   └── Images/             # Training images folder
└── test/
    ├── test.clean.npz      # Test data with text embeddings
    └── captions.txt        # Test captions
```

### Data Files Description:

  * **train.npz**: Contains `captions/embeddings` (1024-dim text), `images/embeddings` (1536-dim image), `captions/text`, `images/names`.
  * **test.clean.npz**: Contains `captions/ids`, `captions/text`, `captions/embeddings` for inference.

## 🧠 Methodology and Model Evolution

The core task is to learn a translator function $f$ that maps a text embedding $Z_c$ to the image embedding space, creating a predicted embedding $\hat{Z}_i$.

Our key insight, is that this is a **latent space alignment** problem, not a simple regression. The text and image latent spaces are semantically similar but likely rotated or reflected. Therefore, optimizing for **cosine similarity** is far more effective than minimizing Mean Squared Error (MSE).

-----

### Model 1: Baseline

Our initial approach, detailed in `src/pipeline.py`, was a 2-layer MLP.

  * **Architecture:** 2-Layer MLP (1024 $\rightarrow$ 4096 $\rightarrow$ 1536)
  * **Loss Function:** Cosine Similarity Loss (`1 - torch.nn.functional.cosine_similarity`)
  * **Optimizer:** Adam (lr=0.001)
  * **Regularization:** Dropout (0.2)
  * **Epochs:** 20
  * **Result:** This model served as a valuable baseline (~0.62 score in the leaderboard) but was outperformed by a contrastive learning approach.

-----

### Model 2: Final

Our best-performing model is implemented in `residual_model.ipynb`. This model reframes the problem from regression to contrastive alignment, inspired by models like CLIP (Contrastive Language-Image Pre-training).

  * **Architecture: `ResidualMLP`**
    The network uses a residual connection to allow the model to learn modifications to a direct linear projection, which helps stabilize training and improve performance.

    1.  **Input ($Z_c$):** 1024-dim
    2.  **Projection:** A linear layer maps the input $Z_c$ from 1024 to 1536 dim. This serves as the `identity` for the residual connection.
    3.  **Hidden Block:**
          * Linear (1024 $\rightarrow$ 2048)
          * LayerNorm
          * GELU (a smooth activation, outperforming ReLU)
          * Dropout (0.2)
          * Linear (2048 $\rightarrow$ 1536)
          * LayerNorm
    4.  **Output:** The output of the hidden block is added to the `identity` projection: `GELU(hidden_output + identity)`

  * **Loss Function: `InfoNCELoss` (InfoNCE)**
    We switched to an **Information Noise Contrastive Estimation (InfoNCE)** loss. Instead of forcing $\hat{Z}_i$ to be *equal* to $Z_i$ (like MSE or Cosine Loss), InfoNCE "pulls" the correct $(\hat{Z}_i, Z_i)$ pair together while "pushing" $\hat{Z}_i$ away from all other "negative" $Z_j$ embeddings in the batch. This directly optimizes for alignment and is robust to rotational differences between the spaces.

  * **Data Preprocessing:**
    An important step is **L2 Normalization** of all input text and target image embeddings *before* training. This places all vectors on the unit hypersphere, making the InfoNCE loss effective.

## ⚙️ Training Strategy (Final Model)

The final model's training pipeline is optimized for performance and stability on GPU.

  * **Optimizer:** `Adam` (learning rate: $1 \times 10^{-4}$)
  * **Scheduler:** `CosineAnnealingLR` (smoothly decays the learning rate following a cosine-shaped schedule)
  * **Batch Size:** 256
  * **Validation:** 10% validation split for performance monitoring
  * **Key Features:**
      * **Mixed Precision Training:** Uses `torch.amp.GradScaler` for faster (up to 2x) training and reduced VRAM usage on GPUs.
      * **Early Stopping:** Monitors the validation loss (patience=3) and saves only the best-performing model, preventing overfitting.
      * **Best Model Checkpointing:** The model with the lowest validation loss is saved as `pipeline_best_model.pth`.

## 🚀 Pipeline Usage

### Main Pipeline (`residual_model.ipynb`)

The notebook containing the complete pipeline for the **final model (Model 2)**.

**What it does:**

1.  **Data Loading:** Loads `train.npz`, L2-normalizes embeddings, and creates a 90/10 train/val split.
2.  **Training:** Trains the `ResidualMLP` using the `InfoNCELoss`.
3.  **Checkpointing:** Automatically saves the best model (`pipeline_best_model.pth`) based on validation loss.
4.  **Inference:** Loads the best model, processes `test.clean.npz`, and generates predictions.
5.  **Submission:** Saves the final predictions to `submission.csv`.

### Baseline Pipeline (`src/pipeline.py`)

The script for the **baseline model (Model 1)**.

```bash
python src/pipeline.py
```

### Verification Script (`src/verif.py`)

A detailed analysis and visualization tool for evaluating model performance.

```bash
python src/verif.py
```

**Features:**

  * Distance analysis (Euclidean, Manhattan, Cosine)
  * Performance comparison vs. random baselines
  * Visualization of results with sample images
  * Statistical metrics and improvement percentages

## 🗂️ Project Structure

```
├── residual_model.ipynb        # Main notebook for final model
├── src/
│   ├── dataset_loader.py       # Data loading utilities
│   ├── pipeline.py             # Main script for baseline model
│   ├── test.py                 # Legacy test script
│   └── verif.py                # Model verification and analysis
├── playground/                 # Experimental scripts
├── models_archive/             # Saved model files
├── submissions_archive/        # Previous submissions
├── data/                       # Dataset (structure described above)
└── requirements.txt            # Python dependencies
```

## 📦 Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Main dependencies:

  * `torch`
  * `numpy`
  * `pandas`
  * `scikit-learn`
  * `tqdm`
  * `PIL` (Pillow)
  * `matplotlib`
