import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import librosa
from transformers import BertTokenizer
torch.cuda.empty_cache()   # clear GPU memory at the start of the script to prevent fragmentation and ensure maximum available memory for training

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class IEMOCAPDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def extract_mfcc(self, path):
        y, sr = librosa.load(path, sr=16000)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # pad / truncate
        max_len = 100
        if mfcc.shape[1] < max_len:
            pad = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad)))  # pad with zeros on the right 
        else:
            mfcc = mfcc[:, :max_len]     # truncate to max_len frames

        return mfcc

    def __getitem__(self, idx):
        row = self.df.iloc[idx]     # Get the row as a Series

        # AUDIO
        mfcc = self.extract_mfcc(row["audio_path"])

        # TEXT
        encoded = tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        return {
            "audio": torch.tensor(mfcc).float(),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(row["label"])
        }