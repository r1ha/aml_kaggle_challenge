import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image
import os

mlp = nn.Sequential(
        nn.Linear(1024, 1536),      # Direct linear mapping from text to image space
        nn.Dropout(0.1)             # Light regularization for single layer
    )

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlp.load_state_dict(torch.load('pipeline_final_model.pth', map_location=device))
mlp.to(device)
mlp.eval()

print(f"Model loaded on device: {device}")

# Load training data
train_data = np.load("data/train/train.npz")

# Extract data
text_embeddings = train_data["captions/embeddings"]
image_embeddings = train_data["images/embeddings"]
image_names = train_data["images/names"]
caption_texts = train_data["captions/text"]

print(f"\nImages: {len(image_embeddings)}, Captions: {len(text_embeddings)}")
print(f"Each image has {len(text_embeddings) // len(image_embeddings)} captions")

def euclidean_distance(a, b):
    """Calculate Euclidean distance"""
    return np.linalg.norm(a - b)

def manhattan_distance(a, b):
    """Calculate Manhattan (L1) distance"""
    return np.sum(np.abs(a - b))

def cosine_distance(a, b):
    """Calculate Cosine distance (1 - cosine_similarity)"""
    return 1 - cosine_similarity([a], [b])[0][0]

# Sample analysis: first 3 images (15 captions total, 5 per image)
n_images_sample = 3
n_captions_per_image = 5
sample_indices = list(range(n_images_sample * n_captions_per_image))

print("\n=== Distance Analysis ===")
print("Format: [Caption_ID] Image_Name | Caption | Euclidean | Manhattan | Cosine | vs Random")
print("-" * 100)

# Set random seed for reproducible random comparisons
np.random.seed(42)

results = []
for i in sample_indices:
    # Get corresponding image index (5 captions per image)
    image_idx = i // n_captions_per_image
    
    # Get embeddings and data
    caption_embedding = torch.tensor(text_embeddings[i], dtype=torch.float32).to(device)
    true_image_embedding = image_embeddings[image_idx]
    image_name = image_names[image_idx]
    caption_text = caption_texts[i][:50] + "..." if len(caption_texts[i]) > 50 else caption_texts[i]
    
    # Get prediction from MLP
    with torch.no_grad():
        predicted_embedding = mlp(caption_embedding.unsqueeze(0)).cpu().numpy()[0]
    
    # Calculate distances with true image
    eucl_dist = euclidean_distance(predicted_embedding, true_image_embedding)
    manh_dist = manhattan_distance(predicted_embedding, true_image_embedding)
    cos_dist = cosine_distance(predicted_embedding, true_image_embedding)
    
    # Calculate distance with random image embedding for comparison
    random_image_idx = np.random.randint(0, len(image_embeddings))
    while random_image_idx == image_idx:  # Ensure it's different
        random_image_idx = np.random.randint(0, len(image_embeddings))
    
    random_image_embedding = image_embeddings[random_image_idx]
    eucl_dist_random = euclidean_distance(predicted_embedding, random_image_embedding)
    cos_dist_random = cosine_distance(predicted_embedding, random_image_embedding)
    
    # Store results
    results.append({
        'caption_idx': i,
        'image_idx': image_idx,
        'image_name': image_name,
        'caption_text': caption_text,
        'eucl_true': eucl_dist,
        'manh_true': manh_dist,
        'cos_true': cos_dist,
        'eucl_random': eucl_dist_random,
        'cos_random': cos_dist_random,
        'improvement_eucl': (eucl_dist_random - eucl_dist) / eucl_dist_random * 100,
        'improvement_cos': (cos_dist_random - cos_dist) / cos_dist_random * 100
    })
    
    print(f"[{i:2d}] {image_name} | {caption_text[:30]:30s} | {eucl_dist:6.2f} | {manh_dist:8.1f} | {cos_dist:6.4f} | R:{eucl_dist_random:6.2f}")

# Summary statistics
print("\n=== Summary Statistics ===")
eucl_true = [r['eucl_true'] for r in results]
cos_true = [r['cos_true'] for r in results]
eucl_random = [r['eucl_random'] for r in results]
cos_random = [r['cos_random'] for r in results]

print(f"Euclidean Distance - True pairs:   {np.mean(eucl_true):.2f} ± {np.std(eucl_true):.2f}")
print(f"Euclidean Distance - Random pairs: {np.mean(eucl_random):.2f} ± {np.std(eucl_random):.2f}")
print(f"Cosine Distance - True pairs:      {np.mean(cos_true):.4f} ± {np.std(cos_true):.4f}")
print(f"Cosine Distance - Random pairs:    {np.mean(cos_random):.4f} ± {np.std(cos_random):.4f}")

improvement_eucl = np.mean([r['improvement_eucl'] for r in results])
improvement_cos = np.mean([r['improvement_cos'] for r in results])
print(f"\nAverage improvement over random:")
print(f"Euclidean: {improvement_eucl:.1f}%")
print(f"Cosine: {improvement_cos:.1f}%")

# Visualize images and results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('MLP Prediction Analysis: Images and Distance Metrics', fontsize=16)

# Display images
for idx in range(n_images_sample):
    ax = axes[0, idx]
    image_name = results[idx * n_captions_per_image]['image_name']
    image_path = f"data/train/Images/{image_name}"
    
    if os.path.exists(image_path):
        img = Image.open(image_path)
        ax.imshow(img)
        ax.set_title(f"Image: {image_name}\n(5 captions analyzed)", fontsize=10)
    else:
        ax.text(0.5, 0.5, f"Image not found:\n{image_name}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Missing: {image_name}", fontsize=10)
    
    ax.axis('off')

# Distance comparison plots
ax_eucl = axes[1, 0]
ax_eucl.scatter(range(len(eucl_true)), eucl_true, label='True pairs', alpha=0.7, color='blue', s=50)
ax_eucl.scatter(range(len(eucl_random)), eucl_random, label='Random pairs', alpha=0.7, color='red', s=50)
ax_eucl.set_xlabel('Caption Index')
ax_eucl.set_ylabel('Euclidean Distance')
ax_eucl.set_title('Euclidean Distance')
ax_eucl.legend()
ax_eucl.grid(True, alpha=0.3)

ax_cos = axes[1, 1]
ax_cos.scatter(range(len(cos_true)), cos_true, label='True pairs', alpha=0.7, color='blue', s=50)
ax_cos.scatter(range(len(cos_random)), cos_random, label='Random pairs', alpha=0.7, color='red', s=50)
ax_cos.set_xlabel('Caption Index')
ax_cos.set_ylabel('Cosine Distance')
ax_cos.set_title('Cosine Distance')
ax_cos.legend()
ax_cos.grid(True, alpha=0.3)

# Improvement percentages
ax_imp = axes[1, 2]
improvements_eucl = [r['improvement_eucl'] for r in results]
improvements_cos = [r['improvement_cos'] for r in results]
x_pos = np.arange(len(improvements_eucl))
width = 0.35

ax_imp.bar(x_pos - width/2, improvements_eucl, width, label='Euclidean', alpha=0.7)
ax_imp.bar(x_pos + width/2, improvements_cos, width, label='Cosine', alpha=0.7)
ax_imp.set_xlabel('Caption Index')
ax_imp.set_ylabel('Improvement (%)')
ax_imp.set_title('Improvement over Random')
ax_imp.legend()
ax_imp.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distance_analysis_with_images.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved as 'distance_analysis_with_images.png'")

# Detailed results for each caption
print("\n=== Detailed Results ===")
for r in results:
    print(f"\nCaption {r['caption_idx']}: \"{r['caption_text']}\"")
    print(f"  Image: {r['image_name']}")
    print(f"  Distances to TRUE image:   Eucl={r['eucl_true']:.2f}, Cos={r['cos_true']:.4f}")
    print(f"  Distances to RANDOM image: Eucl={r['eucl_random']:.2f}, Cos={r['cos_random']:.4f}")
    print(f"  Improvement: Eucl={r['improvement_eucl']:.1f}%, Cos={r['improvement_cos']:.1f}%")




