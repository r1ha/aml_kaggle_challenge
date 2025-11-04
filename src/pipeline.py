import os
import numpy as np
import torch
import pandas as pd
from dataset_loader import create_dataloader
from config import *  # Import all constants and paths
from model import TextToImageMLP, CosineSimilarityLoss  # Import from model.py


def main():
    print("=== TRAINING PHASE ===")

    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Fix seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load training data
    print(f"Loading data from: {TRAIN_PATH}")
    data = np.load(TRAIN_PATH)
    device = DEVICE
    print(f"Using device: {device}")

    # Prepare tensors on the device
    text_embeddings = torch.tensor(data["captions/embeddings"], dtype=torch.float32, device=device)
    image_embeddings_small = torch.tensor(data["images/embeddings"], dtype=torch.float32, device=device)
    image_embeddings = image_embeddings_small.repeat_interleave(5, dim=0)

    # Create dataloader
    train_loader = create_dataloader(text_embeddings, image_embeddings, batch_size=BATCH_SIZE)

    # Define model, loss, optimizer
    mlp = TextToImageMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT).to(device)
    loss_function = CosineSimilarityLoss().to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"Training on {device}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print("CUDA not available (CPU mode)")

    # === Training loop ===
    for epoch in range(EPOCHS):
        mlp.train()
        epoch_losses = []

        for text_batch, image_batch in train_loader:
            optimizer.zero_grad()
            output = mlp(text_batch)
            loss = loss_function(output, image_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.6f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': mlp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
    torch.save(mlp.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")

    # === Quick training evaluation ===
    mlp.eval()
    eval_losses = []
    with torch.no_grad():
        for i, (text_batch, image_batch) in enumerate(train_loader):
            if i >= EVAL_BATCHES:
                break
            output = mlp(text_batch)
            loss = loss_function(output, image_batch)
            eval_losses.append(loss.item())
    print(f"Training Eval Avg Loss: {np.mean(eval_losses):.6f}")

    # === TESTING PHASE ===
    print("\n=== TESTING PHASE ===")
    print(f"Loading test data from: {TEST_PATH}")
    test_data = np.load(TEST_PATH)
    caption_ids = test_data["captions/ids"]
    caption_texts = test_data["captions/text"]
    test_text_embeddings = torch.tensor(test_data["captions/embeddings"], dtype=torch.float32, device=device)

    print(f"Loaded {len(caption_ids)} test captions")

    # Inference
    predictions = []
    with torch.no_grad():
        for i in range(0, len(test_text_embeddings), INFER_BATCH_SIZE):
            batch = test_text_embeddings[i:i+INFER_BATCH_SIZE]
            pred_batch = mlp(batch)
            predictions.append(pred_batch.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)

    # === Evaluation on train subset ===
    n_eval = min(N_EVAL_SAMPLES, len(text_embeddings))
    with torch.no_grad():
        eval_pred = mlp(text_embeddings[:n_eval])
        eval_true = image_embeddings[:n_eval]
        cosine_scores = torch.nn.functional.cosine_similarity(eval_pred, eval_true, dim=1).cpu().numpy()

    print(f"Average Cosine Similarity: {np.mean(cosine_scores):.4f} ± {np.std(cosine_scores):.4f}")

    # === Submission ===
    submission_data = [
        {"id": caption_ids[i], "embedding": predictions[i].tolist()}
        for i in range(len(caption_ids))
    ]
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH} ({len(submission_df)} rows)")


if __name__ == "__main__":
    main()
