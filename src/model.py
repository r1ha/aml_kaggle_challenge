# src/model.py
import torch
import torch.nn as nn

class TextToImageMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=1536, dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class CosineSimilarityLoss(nn.Module):
    """Custom loss to maximize cosine similarity between text and image embeddings."""
    def __init__(self):
        super().__init__()
        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def forward(self, pred, target):
        return 1 - self.cosine_sim(pred, target).mean()
