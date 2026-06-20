import whisper
import os
import torch
device_cpu = torch.device("cpu")  # Define a separate device for CPU to load audio and text models, which can help save GPU memory since these models are less computationally intensive compared to the video model.

def transcribe_audio(audio_path, output_txt_path):
    model = whisper.load_model("base")
    with torch.no_grad():
        result = model.transcribe(audio_path, device="cpu")

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"Transcript saved to {output_txt_path}")
