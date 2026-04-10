# video_dataset.py: This file contains the implementation of the VideoFrameDataset class, 
# which is a custom PyTorch Dataset for loading and processing video frames for training the video model. 
# The dataset is structured to read frames from the processed dataset directory, apply necessary transformations, and provide samples consisting of sequences of frames and their corresponding labels for training.
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms

label_map = {
    "baseline": 0,
    "scripted": 1,
    "cognitive_load": 2,
    "time_pressure": 3,
    "controlled_expression": 4
}
# VideoFrameDataset: A custom PyTorch Dataset class that loads video frames from the processed dataset, 
# applies necessary transformations, and provides samples for training the video model. 
# Each sample consists of a sequence of frames (as a tensor) and the corresponding label for the video.

class VideoFrameDataset(Dataset):
    def __init__(self, frames_root="processed/frames", seq_len=16):
        self.frames_root = frames_root
        self.seq_len = seq_len
        self.samples = []

        for condition in os.listdir(frames_root):
            condition_path = os.path.join(frames_root, condition)

            if os.path.isdir(condition_path):
                for video_folder in os.listdir(condition_path):
                    video_path = os.path.join(condition_path, video_folder)
                    if os.path.isdir(video_path):
                        self.samples.append((video_path, label_map[condition]))
        
        # Define a transformation pipeline for the video frames. This includes resizing the frames to 224x224 pixels,
        # converting them to tensors, and normalizing them using the mean and standard deviation values commonly used for pretrained models like ResNet. 
        # The transforms.Compose function is used to chain these transformations together, so that they can be applied sequentially to each frame when loading the data.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # The __len__ method returns the total number of samples in the dataset, which is determined by the length of the self.samples list. 
    # This allows PyTorch to know how many samples are available for training or evaluation when using this dataset with a DataLoader.
    def __len__(self):
        return len(self.samples)

    # The sample_frames method is a helper function that takes a list of frame file paths and samples a fixed number of frames (seq_len) from the list.
    # If the total number of frames is greater than or equal to seq_len, it uses torch.linspace to generate evenly spaced indices to select frames from the list.
    # If the total number of frames is less than seq_len, it randomly samples indices from the available frames using torch.randint. 
    # This method ensures that we get a consistent number of frames for each video, which is important for training the video model, while also handling cases where some videos may have fewer frames than the desired sequence length.
    def sample_frames(self, frame_files):
        total_frames = len(frame_files)

        if total_frames >= self.seq_len:
            indices = torch.linspace(0, total_frames - 1, self.seq_len).long()
        else:
            indices = torch.randint(0, total_frames, (self.seq_len,))

        return [frame_files[i] for i in indices]

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frame_files = sorted(os.listdir(video_path))
        frame_files = [os.path.join(video_path, f) for f in frame_files if f.endswith(".jpg")]

        selected_frames = self.sample_frames(frame_files)

        frames = []
        for frame_path in selected_frames:
            img = Image.open(frame_path).convert("RGB")
            img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames)  # shape: (seq_len, 3, 224, 224)

        return frames, torch.tensor(label)