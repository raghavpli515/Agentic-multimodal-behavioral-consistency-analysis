import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from train.fusion_dataset import FusionDataset
from models.video_model import VideoModel
from models.audio_model import AudioModel
from models.fusion_model import FusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset
dataset = FusionDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Load unimodal models
video_model = VideoModel().to(device)
audio_model = AudioModel().to(device)

# Freeze unimodal models
for param in video_model.parameters():
    param.requires_grad = False

for param in audio_model.parameters():
    param.requires_grad = False
#The FusionModel class is a PyTorch neural network module that combines the embeddings from the video and audio models to perform multimodal classification. The constructor takes the dimensions of the video and audio embeddings (video_dim and audio_dim) and the number of output classes (num_classes) as arguments. It defines a fusion network that consists of fully connected layers with ReLU activations and dropout for regularization, ultimately producing a final output for classification. The forward method takes the video and audio embeddings as input, concatenates them along the feature dimension, and passes the combined representation through the fusion network to produce the final output for classification. This allows the model to learn how to effectively combine information from both modalities (video and audio) to make predictions.
fusion_model = FusionModel(
    video_dim=video_model.embedding_dim,
    audio_dim=audio_model.embedding_dim
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-3)

epochs = 15

for epoch in range(epochs):

    fusion_model.train()
    train_correct = 0
    train_total = 0

    for frames, mfcc, labels in train_loader:
        frames = frames.to(device)
        mfcc = mfcc.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            video_embed = video_model(frames, return_embedding=True)
            audio_embed = audio_model(mfcc, return_embedding=True)

        outputs = fusion_model(video_embed, audio_embed)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total

    # Validation
    fusion_model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for frames, mfcc, labels in val_loader:
            frames = frames.to(device)
            mfcc = mfcc.to(device)
            labels = labels.to(device)

            video_embed = video_model(frames, return_embedding=True)
            audio_embed = audio_model(mfcc, return_embedding=True)

            outputs = fusion_model(video_embed, audio_embed)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Acc: {train_acc:.2f}%")
    print(f"Val Acc: {val_acc:.2f}%")
    print("-" * 40)

torch.save(fusion_model.state_dict(), "models_saved/fusion_model.pt")
print("Fusion model saved.")