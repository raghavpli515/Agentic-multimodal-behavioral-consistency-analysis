import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

label_map = {
    "baseline": 0,
    "scripted": 1,
    "cognitive_load": 2,
    "time_pressure": 3,
    "controlled_expression": 4
}

class AudioDataset(Dataset):
    # AudioDataset: A custom PyTorch Dataset class that loads audio files from the processed dataset, extracts MFCC features, and provides samples for training the audio model. Each sample consists of a tensor of MFCC features and the corresponding label for the audio clip.
    def __init__(self, audio_root="processed/audio", max_len=200, n_mfcc=40):    # max_len is the maximum number of time steps for the MFCC features, and n_mfcc is the number of MFCC coefficients to extract. These parameters help to ensure that all audio samples have a consistent shape for training the audio model.
        self.audio_root = audio_root
        self.max_len = max_len
        self.n_mfcc = n_mfcc
        self.samples = []

        for condition in os.listdir(audio_root):
            condition_path = os.path.join(audio_root, condition)

            if os.path.isdir(condition_path):
                for file in os.listdir(condition_path):
                    if file.endswith(".wav"):
                        file_path = os.path.join(condition_path, file)
                        self.samples.append((file_path, label_map[condition]))

    def __len__(self):
        return len(self.samples)

    # The pad_or_truncate method is a helper function that takes an MFCC feature matrix and either pads it with zeros or truncates it to ensure that it has a consistent number of time steps (max_len).
    # If the number of time steps in the MFCC matrix is greater than max_len, it truncates the matrix to keep only the first max_len time steps. If the number of time steps is less than max_len, it pads the matrix with zeros along the time dimension until it reaches max_len. This ensures that all MFCC feature matrices have the same shape, which is necessary for batching and training the audio model.
    def pad_or_truncate(self, mfcc):
        if mfcc.shape[1] > self.max_len:
            return mfcc[:, :self.max_len]
        else:
            pad_width = self.max_len - mfcc.shape[1]
            return np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

    # The __getitem__ method is responsible for loading an audio file, extracting its MFCC features, 
    # and returning a sample consisting of the MFCC features as a tensor and the corresponding label as a tensor.
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        # Load the audio file using librosa, which returns the audio time series (y) and the sampling rate (sr). The sr=16000 argument resamples the audio to 16 kHz, which is a common sampling rate for speech processing tasks.
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

        mfcc = self.pad_or_truncate(mfcc)
        # Transpose the MFCC matrix to have the shape (time, n_mfcc) and convert it to a PyTorch tensor of type float32. This format is suitable for input into the audio model, which expects a sequence of feature vectors over time.
        mfcc = torch.tensor(mfcc.T, dtype=torch.float32)  # shape: (time, n_mfcc)

        return mfcc, torch.tensor(label)