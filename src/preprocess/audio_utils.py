# src/preprocess/audio_utils.py

import subprocess
import librosa
import numpy as np
import tempfile
import os


# ==========================================
# AUDIO EXTRACTION
# ==========================================
def extract_audio_segment(video_path, start_time, end_time):
    """
    Extract audio segment from video using FFmpeg.
    Returns path to temporary WAV file.
    """

    duration = end_time - start_time

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    temp_audio_path = temp_file.name
    temp_file.close()

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ss", str(start_time),
        "-t", str(duration),
        "-ac", "1",          # mono
        "-ar", "16000",      # 16kHz
        temp_audio_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg extraction failed:\n"
            f"{result.stderr.decode(errors='ignore')}"
        )

    if not os.path.exists(temp_audio_path):
        raise RuntimeError(
            "FFmpeg output audio file not created."
        )

    if os.path.getsize(temp_audio_path) == 0:
        raise RuntimeError(
            "Extracted audio file is empty."
        )

    return temp_audio_path


# ==========================================
# MFCC COMPUTATION
# ==========================================
def compute_mfcc(
    audio_path,
    n_mfcc=40,
    max_len=100
):
    """
    Convert audio into fixed-length MFCC features.

    Output shape:
        (40, 100)

    Returns:
        numpy.ndarray(float32)
    """

    try:

        # ----------------------------------
        # LOAD AUDIO
        # ----------------------------------
        y, sr = librosa.load(
            audio_path,
            sr=16000
        )

        print("\n===== AUDIO DEBUG =====")
        print("Audio Path:", audio_path)
        print("Sample Rate:", sr)
        print("Audio Samples:", len(y))

        # ----------------------------------
        # HANDLE EMPTY AUDIO
        # ----------------------------------
        if len(y) == 0:
            print("[WARNING] Empty audio detected.")

            return np.zeros(
                (n_mfcc, max_len),
                dtype=np.float32
            )

        # ----------------------------------
        # HANDLE SILENT AUDIO
        # ----------------------------------
        max_amp = np.max(np.abs(y))

        print("Max Amplitude:", max_amp)

        if max_amp < 1e-6:
            print("[WARNING] Silent audio detected.")

            return np.zeros(
                (n_mfcc, max_len),
                dtype=np.float32
            )

        # ----------------------------------
        # MFCC EXTRACTION
        # ----------------------------------
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc
        )

        print("Raw MFCC Shape:", mfcc.shape)

        # ----------------------------------
        # CHECK INVALID VALUES
        # ----------------------------------
        print(
            "MFCC Contains NaN:",
            np.isnan(mfcc).any()
        )

        print(
            "MFCC Contains Inf:",
            np.isinf(mfcc).any()
        )

        mfcc = np.nan_to_num(
            mfcc,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ----------------------------------
        # FIXED LENGTH
        # ----------------------------------
        if mfcc.shape[1] < max_len:

            pad_width = max_len - mfcc.shape[1]

            mfcc = np.pad(
                mfcc,
                ((0, 0), (0, pad_width)),
                mode="constant"
            )

        else:

            mfcc = mfcc[:, :max_len]

        # ----------------------------------
        # FINAL SAFETY
        # ----------------------------------
        mfcc = mfcc.astype(np.float32)

        print("Final MFCC Shape:", mfcc.shape)
        print("Final MFCC Min:", np.min(mfcc))
        print("Final MFCC Max:", np.max(mfcc))

        return mfcc

    except Exception as e:

        print(
            f"[WARNING] MFCC extraction failed: {e}"
        )

        return np.zeros(
            (n_mfcc, max_len),
            dtype=np.float32
        )