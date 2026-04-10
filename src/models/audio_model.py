import torch
import torch.nn as nn
# This model is designed for audio classification tasks, such as speech emotion recognition.
class AudioModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, num_classes=5):
        super(AudioModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.embedding_dim = hidden_dim * 2 # Since the LSTM is bidirectional, the output dimension is hidden_dim * 2 (one for the forward direction and one for the backward direction).

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_embedding=False):
        lstm_out, _ = self.lstm(x)
        final_output = lstm_out[:, -1, :]

        if return_embedding:
            return final_output

        return self.classifier(final_output)