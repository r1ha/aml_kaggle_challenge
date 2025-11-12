import torch
from torch.utils.data import Dataset, DataLoader


class TextImageEmbeddingDataset(Dataset):
    def __init__(self, text_embeddings, image_embeddings, device=None):
        """
        Args:
            text_embeddings: torch.Tensor (N, 1024)
            image_embeddings: torch.Tensor (N, 1536)
            device: 'cuda' or 'cpu'
        """
        assert len(text_embeddings) == len(
            image_embeddings), "Sizes must match"
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, idx):
        # Returns a pair (text embedding, image embedding)
        text_emb = self.text_embeddings[idx]
        img_emb = self.image_embeddings[idx]
        return text_emb.to(self.device), img_emb.to(self.device)


def create_dataloader(text_tensor, image_tensor, batch_size=64, shuffle=True, num_workers=0):
    """
    Create a ready to use PyTorch DataLoader for text-image embedding pairs.
    num_workers: Number of subprocesses to use for data loading.
    0 means that the data will be loaded in the main process
    1 or more means that data loading will be done in separate subprocesses.
    """
    dataset = TextImageEmbeddingDataset(text_tensor, image_tensor)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers)
    return loader
