import subprocess
import os
import librosa
import numpy as np
import tempfile


def extract_audio_segment(video_path, start_time, end_time):
    """
    Fast audio extraction using FFmpeg (no full video reload)
    """
    
    duration = end_time - start_time

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav" , delete=False)  #delete False to keep file after closing
    temp_audio_path = temp_file.name
    temp_file.close()

    command = [         # FFmpeg command to extract audio segment 
        "ffmpeg",       # -y to overwrite without asking
        "-y",           # input file
        "-i", video_path,  # output options
        "-ss",  str(start_time),   # start time
        "-t", str(duration),     # duration
        "-ac", "1",         # mono audio
        "-ar", "16000",     # sample rate
        temp_audio_path     # output file
    ]
    
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)   # suppress FFmpeg output, stderr=subprocess.DEVNULL to suppress error messages, stdout=subprocess.DEVNULL to suppress standard output
    return temp_audio_path


def compute_mfcc(audio_path, n_mfcc=40, max_len=100):
    """
    Convert audio to MFCC features
    """

    y, sr = librosa.load(audio_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)    # compute MFCC features, mfcc shape is (n_mfcc, time_frames),n_mfcc is the number of MFCC features to extract, time_frames is the number of frames in the audio segment

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))   # pad with zeros to ensure consistent length, pad_width is the number of frames to pad, (0, 0) means no padding for the n_mfcc dimension, (0, pad_width) means pad with zeros at the end of the time_frames dimension
    else:
        mfcc = mfcc[:, :max_len]   # truncate to max_len frames if too long, keep only the first max_len frames in the time_frames dimension

    return mfcc