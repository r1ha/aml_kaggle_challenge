from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dataset_loader import create_dataloader

def main():
    print("=== TRAINING PHASE ===")
    
    # Load training data and move directly to GPU
    data = np.load("data/train/train.npz")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_embeddings = torch.tensor(data["captions/embeddings"], dtype=torch.float32, device=device)
    image_embeddings_small = torch.tensor(data["images/embeddings"], dtype=torch.float32, device=device)
    image_embeddings = image_embeddings_small.repeat_interleave(5, dim=0)
    
    train_loader = create_dataloader(text_embeddings, image_embeddings, batch_size=64)
    
    # Two-layer MLP with semantic expansion architecture
    mlp = nn.Sequential(
        nn.Linear(1024, 2048),      # First layer: expand text embedding to larger semantic space
        nn.ReLU(),                  # Non-linear activation for feature transformation
        nn.Dropout(0.2),            # Regularization to prevent overfitting
        nn.Linear(2048, 1536)       # Second layer: map semantic features to image embedding space
    )
    
    class CosineSimilarityLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.cosine_sim = nn.CosineSimilarity(dim=1)
        
        def forward(self, pred, target):
            return 1 - self.cosine_sim(pred, target).mean()
    
    loss_function = CosineSimilarityLoss()
    # Reduced learning rate for stable two-layer training
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training parameters optimized for two-layer architecture
    num_epochs = 20  # Adjusted for two-layer convergence
    mlp.to(device)
    loss_function.to(device)
    
    print(f"Training on device: {device}")
    
    # GPU verification and memory info
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available")
    
    # Training loop
    for epoch in range(num_epochs):
        mlp.train()
        epoch_losses = []
        
        for batch_idx, (text_batch, image_batch) in enumerate(train_loader):
            # Data is already on GPU from dataloader
            optimizer.zero_grad()
            output = mlp(text_batch)
            loss = loss_function(output, image_batch)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Calculate and print epoch average loss
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
        
        # Save model checkpoint every 5 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': mlp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'pipeline_model_checkpoint_epoch_{epoch+1}.pth')
            print(f"Model saved at epoch {epoch+1}")
    
    # Final model save
    torch.save(mlp.state_dict(), 'pipeline_final_model.pth')
    print("Final model saved as 'pipeline_final_model.pth'")
    
    # Evaluation on training data
    mlp.eval()
    eval_losses = []
    with torch.no_grad():
        for i, (text_batch, image_batch) in enumerate(train_loader):
            if i >= 5:  # Evaluate on first 5 batches only
                break
            # Data is already on GPU from dataloader
            output = mlp(text_batch)
            loss = loss_function(output, image_batch)
            eval_losses.append(loss.item())
    
    print(f"Training Evaluation Average Loss: {np.mean(eval_losses):.6f}")
    
    print("\n=== TESTING PHASE ===")
    
    # Load test data
    test_data = np.load("data/test/test.clean.npz")
    
    # Extract test data and move directly to GPU
    caption_ids = test_data["captions/ids"]
    caption_texts = test_data["captions/text"]
    test_text_embeddings = torch.tensor(test_data["captions/embeddings"], dtype=torch.float32, device=device)
    
    print(f"Loaded {len(caption_ids)} test captions")
    print(f"Model loaded on device: {device}")
    
    # Generate predictions (batch processing for better GPU utilization)
    predictions = []
    batch_size = 256  # Larger batch size for inference
    with torch.no_grad():
        for i in range(0, len(test_text_embeddings), batch_size):
            batch = test_text_embeddings[i:i+batch_size]
            predicted_batch = mlp(batch)
            predictions.append(predicted_batch.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
    
    # Evaluate model with cosine similarity on training data
    print("\n=== Model Evaluation ===")
    
    # Get predictions on training data (first 500 samples to avoid memory issues)
    n_eval = min(500, len(text_embeddings))
    with torch.no_grad():
        eval_text = text_embeddings[:n_eval]  # Already on GPU
        eval_predicted = mlp(eval_text)  # Keep on GPU for cosine calculation
        eval_true = image_embeddings[:n_eval]  # Already on GPU
    
    # Calculate cosine similarities on GPU (much faster than CPU)
    with torch.no_grad():
        cosine_sim_gpu = torch.nn.functional.cosine_similarity(eval_predicted, eval_true, dim=1)
        cosine_scores = cosine_sim_gpu.cpu().numpy()  # Only transfer final result
    
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
    
    print(f"Submission saved with {len(submission_df)} rows")
    print(f"Sample submission format:")
    print(submission_df.head(2))

if __name__ == "__main__":
    main()