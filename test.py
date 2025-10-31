from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# Load test data
data = np.load("data/test/test.clean.npz")

# Extract test data
caption_ids = data["captions/ids"]
caption_texts = data["captions/text"]
text_embeddings = torch.tensor(data["captions/embeddings"], dtype=torch.float32)

print(f"\nLoaded {len(caption_ids)} test captions")

# Recreate the same MLP architecture from perceptron.py
mlp = nn.Sequential(
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1536)
)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlp.load_state_dict(torch.load('final_model.pth', map_location=device))
mlp.to(device)
mlp.eval()

print(f"Model loaded on device: {device}")

# Generate predictions
predictions = []
with torch.no_grad():
    text_embeddings = text_embeddings.to(device)
    predicted_image_embeddings = mlp(text_embeddings)
    predictions = predicted_image_embeddings.cpu().numpy()

# Evaluate model with cosine similarity on training data
from sklearn.metrics.pairwise import cosine_similarity
print("\n=== Model Evaluation ===")

# Load training data for evaluation
train_data = np.load("data/train/train.npz")
train_text_embeddings = torch.tensor(train_data["captions/embeddings"], dtype=torch.float32)
train_image_embeddings_small = torch.tensor(train_data["images/embeddings"], dtype=torch.float32)
train_image_embeddings = train_image_embeddings_small.repeat_interleave(5, dim=0)

# Get predictions on training data (first 500 samples to avoid memory issues)
n_eval = min(500, len(train_text_embeddings))
with torch.no_grad():
    eval_text = train_text_embeddings[:n_eval].to(device)
    eval_predicted = mlp(eval_text).cpu().numpy()
    eval_true = train_image_embeddings[:n_eval].numpy()

# Calculate cosine similarities
cosine_scores = []
for i in range(n_eval):
    # Cosine similarity between predicted and true embedding
    cos_sim = cosine_similarity([eval_predicted[i]], [eval_true[i]])[0][0]
    cosine_scores.append(cos_sim)

avg_cosine = np.mean(cosine_scores)
print(f"Average Cosine Similarity on {n_eval} training samples: {avg_cosine:.4f}")
print(f"Cosine Similarity std: {np.std(cosine_scores):.4f}")
print(f"Cosine Similarity range: [{np.min(cosine_scores):.4f}, {np.max(cosine_scores):.4f}]")

# Create submission CSV
submission_data = []
for i, caption_id in enumerate(caption_ids):
    # Convert embedding array to proper list format for Kaggle
    embedding_list = predictions[i].tolist()  # Convert numpy array to Python list
    submission_data.append({
        'id': caption_id,
        'embedding': embedding_list  # Note: 'embedding' not 'embeddings'
    })

# Create DataFrame and save to CSV
submission_df = pd.DataFrame(submission_data)
submission_df.to_csv('submission.csv', index=False, quoting=0)  # quoting=0 removes quotes around arrays

print(f"Submission saved to 'submission.csv' with {len(submission_df)} rows")
print(f"Sample submission format:")
print(submission_df.head(2))