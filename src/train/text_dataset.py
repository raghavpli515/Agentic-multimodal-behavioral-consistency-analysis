import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

label_map = {
    "baseline": 0,
    "scripted": 1,
    "cognitive_load": 2,
    "time_pressure": 3,
    "controlled_expression": 4
}

class TextDataset(Dataset):
    def __init__(self, transcript_root="processed/transcripts", max_len=256):   #max_len is the maximum sequence length for the BERT tokenizer. It determines how many tokens will be included in each input sample. If a transcript has more tokens than max_len, it will be truncated to fit within this limit. If it has fewer tokens, it will be padded with special padding tokens to ensure that all input samples have the same length, which is necessary for batching and training the text model.
        self.transcript_root = transcript_root
        self.max_len = max_len
        self.samples = []

        for condition in os.listdir(transcript_root):
            condition_path = os.path.join(transcript_root, condition)

            if os.path.isdir(condition_path):
                for file in os.listdir(condition_path):
                    if file.endswith(".txt"):
                        file_path = os.path.join(condition_path, file)
                        self.samples.append((file_path, label_map[condition]))

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  #The BERT tokenizer is used to convert the raw text transcripts into token IDs that can be input into a BERT-based text model. The "bert-base-uncased" model is a commonly used pre-trained BERT model that has been trained on a large corpus of English text. By using this tokenizer, we can leverage the powerful language understanding capabilities of BERT for our text classification task. The tokenizer will handle tasks such as tokenization, adding special tokens, and creating attention masks, which are essential for preparing the text data for input into the BERT model.

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        encoding = self.tokenizer(
            text,
            padding="max_length", # The padding="max_length" argument tells the tokenizer to pad all input sequences to the maximum length specified by max_len. This ensures that all input samples have the same length, which is necessary for batching and training the text model. If a transcript has fewer tokens than max_len, it will be padded with special padding tokens until it reaches max_len. If it has more tokens than max_len, it will be truncated to fit within this limit.
            truncation=True, # The truncation=True argument tells the tokenizer to truncate any input sequences that exceed the maximum length specified by max_len. If a transcript has more tokens than max_len, it will be truncated to keep only the first max_len tokens. This is important to ensure that all input samples have a consistent length for batching and training the text model, and to prevent issues with memory and computational efficiency when processing long sequences.
            max_length=self.max_len,
            return_tensors="pt" # The return_tensors="pt" argument tells the tokenizer to return the tokenized output as PyTorch tensors. This is necessary for input into a PyTorch-based text model, such as a BERT model implemented in PyTorch. By returning the tokenized output as tensors, we can easily move the data to the appropriate device (CPU or GPU) and use it for training and inference with our text model.
        )

        input_ids = encoding["input_ids"].squeeze(0) # The encoding["input_ids"] contains the token IDs for the input text, and the .squeeze(0) is used to remove the extra batch dimension added by the tokenizer. Since we are processing one sample at a time in the __getitem__ method, the tokenizer returns a tensor with an extra dimension for the batch size (which is 1 in this case). By using .squeeze(0), we remove this extra dimension and get a tensor of shape (max_len,) that contains the token IDs for the input text, which can then be used as input to the text model.
        attention_mask = encoding["attention_mask"].squeeze(0)

        return input_ids, attention_mask, torch.tensor(label)