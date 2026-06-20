import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import librosa
from transformers import BertTokenizer
import cv2
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class FineTuneDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def extract_mfcc(self, path):
        y, sr = librosa.load(path, sr=16000)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        max_len = 100  # smaller for memory + speed

        if mfcc.shape[1] < max_len:
            pad = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
        else:
            mfcc = mfcc[:, :max_len]

        return mfcc

    def extract_frames(self, video_path, num_frames=6):
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.resize(frame, (112, 112))
            frame = frame / 255.0
            frame = np.transpose(frame, (2, 0, 1))  # C,H,W

            frames.append(frame)

        cap.release()

        frames = np.array(frames)

        # Pad if needed
        if len(frames) < num_frames:
            pad = num_frames - len(frames)
            frames = np.pad(frames, ((0, pad), (0,0), (0,0), (0,0)))

        return frames    
    
    
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # =========================
        # AUDIO
        # =========================
        mfcc = self.extract_mfcc(row["audio_path"])

        # =========================
        # TEXT
        # =========================
        encoded = tokenizer(
            str(row["text"]),
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        # =========================
        # VIDEO
        # =========================
        video = self.extract_frames(row["video_path"])

        # =========================
        # OUTPUT
        # =========================
        return {
            "audio": torch.tensor(mfcc).float(),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["label"])),
            "video": torch.tensor(video).float(),

        }
    
