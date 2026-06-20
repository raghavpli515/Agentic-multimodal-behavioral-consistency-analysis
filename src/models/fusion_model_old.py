import torch
import torch.nn as nn
#The FusionModel class is a PyTorch neural network module that combines the embeddings from the video and audio models to perform multimodal classification.
class FusionModel(nn.Module):
    def __init__(self, video_dim, audio_dim, num_classes=5):  # The constructor takes the dimensions of the video and audio embeddings (video_dim and audio_dim) and the number of output classes (num_classes) as arguments. It defines a fusion network that consists of fully connected layers with ReLU activations and dropout for regularization, ultimately producing a final output for classification.
        super(FusionModel, self).__init__()

        self.fusion = nn.Sequential(
            nn.Linear(video_dim + audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    # The forward method takes the video and audio embeddings as input, concatenates them along the feature dimension, and passes the combined representation through the fusion network to produce the final output for classification. This allows the model to learn how to effectively combine information from both modalities (video and audio) to make predictions.
    def forward(self, video_embed, audio_embed):
        combined = torch.cat((video_embed, audio_embed), dim=1) #The torch.cat function is used to concatenate the video and audio embeddings along the feature dimension (dim=1). This creates a single combined representation that contains information from both modalities, which can then be processed by the fusion network to make predictions.The feature dimension is typically the second dimension (dim=1) in a batch of embeddings, where the first dimension (dim=0) represents the batch size. By concatenating along dim=1, we are effectively combining the features from both the video and audio embeddings for each sample in the batch.
        return self.fusion(combined)