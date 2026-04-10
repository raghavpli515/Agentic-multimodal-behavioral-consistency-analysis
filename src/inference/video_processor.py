import os
import cv2
import torch
import whisper
import librosa
import numpy as np
from moviepy import VideoFileClip
from transformers import BertTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


class VideoProcessor:

    def __init__(self):
        self.whisper_model = whisper.load_model("base", device=device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # -----------------------------
    # FRAME EXTRACTION
    # -----------------------------
    def extract_frames(self, video_path, num_frames=16):

        cap = cv2.VideoCapture(video_path)
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames // num_frames, 1)

        count = 0
        while cap.isOpened() and len(frames) < num_frames:

            ret, frame = cap.read()
            if not ret:
                break

            if count % step == 0:
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0  # Normalize pixel values to [0, 1]
                frame = np.transpose(frame, (2, 0, 1))  # Change from HWC to CHW format,where C is the number of channels (3 for RGB), H is height, and W is width. This format is commonly used in PyTorch models.
                frames.append(frame)

            count += 1

        cap.release()

        frames = np.array(frames) 
        frames = torch.tensor(frames).float().unsqueeze(0)  #.tensor converts the numpy array of frames into a PyTorch tensor. .float() ensures that the data type of the tensor is floating-point, which is typically required for input to neural networks. .unsqueeze(0) adds an extra dimension at the beginning of the tensor, effectively creating a batch dimension. This means that if you have 16 frames of shape (3, 224, 224), after unsqueezing, the shape will become (1, 16, 3, 224, 224), where 1 represents the batch size.

        return frames

    # -----------------------------
    # AUDIO EXTRACTION
    # -----------------------------
    def extract_audio_features(self, video_path):

        clip = VideoFileClip(video_path)  
        audio_path = "temp_audio.wav"  

        clip.audio.write_audiofile(audio_path, logger=None)

        audio, sr = librosa.load(audio_path, sr=16000) #librosa.load loads the audio file at the specified path (audio_path) and resamples it to a sample rate of 16000 Hz (sr=16000). The function returns the audio time series as a numpy array (audio) and the sample rate (sr). Resampling to 16000 Hz is common for speech processing tasks, as it provides a good balance between audio quality and computational efficiency.

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  #librosa.feature.mfcc computes the Mel-frequency cepstral coefficients (MFCCs) from the audio time series. The y parameter is the audio time series, sr is the sample rate, and n_mfcc specifies the number of MFCCs to return (in this case, 40). MFCCs are a common feature representation used in audio processing tasks, especially for speech recognition and classification.

        mfcc = mfcc.T # Transpose the MFCC array to have shape (time_steps, n_mfcc). This means that each row of the resulting mfcc array corresponds to a time step, and each column corresponds to one of the 40 MFCC features. Transposing is often done to match the expected input format of machine learning models, where the time dimension is typically the first dimension.

        if mfcc.shape[0] < 200:  
            pad = np.zeros((200 - mfcc.shape[0], 40)) # If the number of time steps in the MFCC array is less than 200, we create a padding array of zeros with shape (200 - mfcc.shape[0], 40). This means that we will add enough rows of zeros to the MFCC array to make it have 200 time steps, while keeping the number of features (40) unchanged. Padding is often used to ensure that all input samples have the same shape, which is necessary for batch processing in machine learning models.
            mfcc = np.vstack((mfcc, pad))  
        else:
            mfcc = mfcc[:200]  

        mfcc = torch.tensor(mfcc).float().unsqueeze(0) 

        os.remove(audio_path)  

        return mfcc

    # -----------------------------
    # TRANSCRIPTION
    # -----------------------------
    def transcribe(self, video_path):

        result = self.whisper_model.transcribe(video_path)
        text = result["text"] # The result of the transcription is stored in a dictionary called result, and the transcribed text can be accessed using the key "text". This means that after the transcription process is complete, you can retrieve the transcribed text from the result dictionary by referencing result["text"].

        encoding = self.tokenizer(  #The tokenizer is used to convert the transcribed text into a format that can be processed by a language model. The text is tokenized, padded to a maximum length of 256 tokens, and truncated if it exceeds that length. The resulting input IDs and attention mask are returned as tensors that can be fed into a model for further processing, such as classification or sentiment analysis.
            text, 
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt" #The return_tensors="pt" argument specifies that the output should be returned as PyTorch tensors. This means that the input IDs and attention mask will be returned as tensors that can be directly used in PyTorch models without needing to convert them from another format.
        )

        return encoding["input_ids"], encoding["attention_mask"]