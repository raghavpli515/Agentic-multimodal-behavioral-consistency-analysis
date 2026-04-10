import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import librosa
import numpy as np
from torchvision import transforms

label_map = {
    "baseline": 0,
    "scripted": 1,
    "cognitive_load": 2,
    "time_pressure": 3,
    "controlled_expression": 4
}
#The input dataset is implemented as a custom PyTorch Dataset class called FusionDataset. 
#This dataset class is responsible for loading both video frames and audio features for each sample, and providing them in a format suitable for training the fusion model. 
#Each sample consists of a sequence of video frames (as tensors), the corresponding MFCC features from the audio, and the label for the sample. 
#The dataset handles the necessary preprocessing steps, such as resizing and normalizing video frames, extracting MFCC features from audio, and ensuring that all samples have consistent shapes for batching during training. 
class FusionDataset(Dataset):
    def __init__(self,
                 frames_root="processed/frames",
                 audio_root="processed/audio",
                 seq_len=16,  # The seq_len parameter specifies the number of video frames to sample for each video clip. If a video has more frames than seq_len, it will sample frames uniformly across the entire video. If a video has fewer frames than seq_len, it will randomly sample frames with replacement to ensure that each sample has the same number of frames for input into the video model.
                 max_audio_len=200,  #The max_audio_len parameter specifies the maximum number of time steps for the MFCC features extracted from the audio. If the MFCC features have more time steps than max_audio_len, they will be truncated to keep only the first max_audio_len time steps. If they have fewer time steps than max_audio_len, they will be padded with zeros along the time dimension until they reach max_audio_len. This ensures that all audio samples have a consistent shape for input into the audio model.
                 n_mfcc=40):   #The n_mfcc parameter specifies the number of MFCC coefficients to extract from the audio. This determines the dimensionality of the audio features that will be input into the audio model. A common choice for n_mfcc is 40, which provides a good balance between capturing relevant audio information and keeping the feature dimensionality manageable for training.

        self.frames_root = frames_root
        self.audio_root = audio_root
        self.seq_len = seq_len
        self.max_audio_len = max_audio_len
        self.n_mfcc = n_mfcc

        self.samples = []

        for condition in os.listdir(frames_root):
            condition_path = os.path.join(frames_root, condition)

            if os.path.isdir(condition_path):
                for video_folder in os.listdir(condition_path):
                    frame_path = os.path.join(condition_path, video_folder)
                    audio_path = os.path.join(audio_root, condition, video_folder + ".wav")

                    if os.path.isdir(frame_path) and os.path.exists(audio_path):
                        self.samples.append((condition, video_folder))  

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def sample_frames(self, frame_files):
        total_frames = len(frame_files)

        if total_frames >= self.seq_len:
            indices = torch.linspace(0, total_frames - 1, self.seq_len).long()
        else:
            indices = torch.randint(0, total_frames, (self.seq_len,))

        return [frame_files[i] for i in indices]

    def pad_or_truncate(self, mfcc):
        if mfcc.shape[1] > self.max_audio_len:
            return mfcc[:, :self.max_audio_len]
        else:
            pad_width = self.max_audio_len - mfcc.shape[1]
            return np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

    def __getitem__(self, idx):
        condition, video_name = self.samples[idx]

        # ---- VIDEO ----
        frame_dir = os.path.join(self.frames_root, condition, video_name)
        frame_files = sorted([
            os.path.join(frame_dir, f)
            for f in os.listdir(frame_dir)
            if f.endswith(".jpg")
        ])

        selected_frames = self.sample_frames(frame_files)

        frames = []
        for frame_path in selected_frames:
            img = Image.open(frame_path).convert("RGB")
            img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames)

        # ---- AUDIO ----
        audio_path = os.path.join(self.audio_root, condition, video_name + ".wav")
        y, sr = librosa.load(audio_path, sr=16000)   
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)  
        mfcc = self.pad_or_truncate(mfcc)
        mfcc = torch.tensor(mfcc.T, dtype=torch.float32)

        label = torch.tensor(label_map[condition])

        return frames, mfcc, label