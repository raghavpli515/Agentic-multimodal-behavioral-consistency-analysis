import os
import pandas as pd
import subprocess
import whisper

ROOT = "D:/PROJECTS/Multimodal_root/dataset"

LABEL_MAP = {
    "baseline": 0,
    "cognitive_load": 1,
    "controlled_expression": 2,
    "scripted": 3,
    "time_pressure": 4
}

# Load Whisper model
whisper_model = whisper.load_model("base")

output = []

for category in os.listdir(ROOT):

    category_path = os.path.join(ROOT, category)

    if not os.path.isdir(category_path):
        continue

    label = LABEL_MAP[category]

    for file in os.listdir(category_path):

        if not file.endswith(".mp4"):
            continue

        video_path = os.path.join(category_path, file)

        # =========================
        # AUDIO EXTRACTION
        # =========================
        audio_path = video_path.replace(".mp4", ".wav")

        if not os.path.exists(audio_path):
            cmd = f'ffmpeg -i "{video_path}" -ar 16000 -ac 1 "{audio_path}"'
            subprocess.call(cmd, shell=True)

        # =========================
        # WHISPER TRANSCRIPTION
        # =========================
        result = whisper_model.transcribe(audio_path)
        text = result["text"].strip().lower()

        output.append({
            "audio_path": audio_path,
            "text": text,
            "label": label
        })

        print(f"Processed: {file}")

df = pd.DataFrame(output)
df.to_csv("personal_dataset.csv", index=False)

print("\n Dataset created:", len(df))