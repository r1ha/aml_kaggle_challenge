from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from dataset_loader import create_dataloader

data = np.load("data/train/train.npz")

text_embeddings = torch.tensor(data["captions/embeddings"], dtype=torch.float32)
image_embeddings_small = torch.tensor(data["images/embeddings"], dtype=torch.float32)
image_embeddings = image_embeddings_small.repeat_interleave(5, dim=0)

train_loader = create_dataloader(text_embeddings, image_embeddings, batch_size=64)

mlp = nn.Sequential(
    nn.Linear(1024, 1024), # Input size matches text embedding size
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1536) # Output size matches image embedding size
)

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_sim = nn.CosineSimilarity(dim=1)
    
    def forward(self, pred, target):
        return 1 - self.cosine_sim(pred, target).mean()

loss_function = CosineSimilarityLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

# Training parameters
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlp.to(device)

print(f"Training on device: {device}")

# Training loop
for epoch in range(num_epochs):
    mlp.train()
    epoch_losses = []
    
    for batch_idx, (text_batch, image_batch) in enumerate(train_loader):
        text_batch, image_batch = text_batch.to(device), image_batch.to(device)
        
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
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': mlp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'model_checkpoint_epoch_{epoch+1}.pth')
        print(f"Model saved at epoch {epoch+1}")

# Final model save
torch.save(mlp.state_dict(), 'final_model.pth')
print("Final model saved as 'final_model.pth'")

# Evaluation on a few batches
mlp.eval()
eval_losses = []
with torch.no_grad():
    for i, (text_batch, image_batch) in enumerate(train_loader):
        if i >= 5:  # Evaluate on first 5 batches only
            break
        text_batch, image_batch = text_batch.to(device), image_batch.to(device)
        output = mlp(text_batch)
        loss = loss_function(output, image_batch)
        eval_losses.append(loss.item())

print(f"Evaluation Average Loss: {np.mean(eval_losses):.6f}")