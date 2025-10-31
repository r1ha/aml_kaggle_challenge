from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image

image_model = AutoModel.from_pretrained('facebook/dinov2-giant')
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant')

image = Image.open("data/train/Images/36979.jpg")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = image_model(**inputs)
image_embedding = outputs.last_hidden_state[:, 0, :]  # token CLS

print(image_embedding.shape)