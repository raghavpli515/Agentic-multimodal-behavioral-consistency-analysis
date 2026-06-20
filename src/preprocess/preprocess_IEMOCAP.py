import os
import pandas as pd
import re

ROOT = "D:/PROJECTS/MMDL_data/IEMOCAP_full_release"

VALID_EMOTIONS = {
    "ang": 0,
    "hap": 1,
    "exc": 1,
    "sad": 2,
    "neu": 3
}

# =============================================
# PARSE EMOTION FILE
# =============================================

emotion_pattern = re.compile(
    r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(Ses\d+.+?)\s+(\w+)"
)

def parse_emotion_file(path):
    data = []

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = emotion_pattern.search(line)
            if not match:
                continue

            start = float(match.group(1))
            end = float(match.group(2))
            utt_id = match.group(3)
            emo = match.group(4)

            if emo not in VALID_EMOTIONS:
                continue

            data.append({
                "utt_id": utt_id,
                "start": start,
                "end": end,
                "label": VALID_EMOTIONS[emo]
            })

    return data


# =============================================
# PARSE TRANSCRIPTIONS
# =============================================

transcription_pattern = re.compile(
    r"(Ses\d+.+?)\s+\[\d+\.\d+-\d+\.\d+\]:\s+(.*)"
)

def parse_transcription(path):
    text_map = {}

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = transcription_pattern.search(line)
            if not match:
                continue

            utt_id = match.group(1)
            text = match.group(2)

            text_map[utt_id] = text.strip()

    return text_map


# =============================================
# MAIN DATASET BUILDER
# =============================================

dataset = []

for session in range(1, 6):

    session_path = os.path.join(ROOT, f"Session{session}")

    emo_dir = os.path.join(session_path, "dialog", "EmoEvaluation")
    trans_dir = os.path.join(session_path, "dialog", "transcriptions")
    video_dir = os.path.join(session_path, "dialog", "avi", "DivX")
    audio_root = os.path.join(session_path, "sentences", "wav")

    print(f"\nProcessing Session{session}...")

    for emo_file in os.listdir(emo_dir):

        if not emo_file.endswith(".txt"):
            continue

        dialog_id = emo_file.replace(".txt", "")

        emo_path = os.path.join(emo_dir, emo_file)
        trans_path = os.path.join(trans_dir, emo_file)
        video_path = os.path.join(video_dir, f"{dialog_id}.avi")

        if not os.path.exists(trans_path):
            continue

        if not os.path.exists(video_path):
            continue

        emo_data = parse_emotion_file(emo_path)
        text_map = parse_transcription(trans_path)

        audio_dir = os.path.join(audio_root, dialog_id)

        if not os.path.exists(audio_dir):
            continue

        for item in emo_data:

            utt_id = item["utt_id"]
            audio_path = os.path.join(audio_dir, f"{utt_id}.wav")

            if not os.path.exists(audio_path):
                continue

            text = text_map.get(utt_id, "")

            # =========================
            #  TEXT CLEANING (ADD HERE)
            # =========================
            text = text.lower()
            text = text.replace("[breathing]", "")
            text = text.replace("--", " ")
            text = text.strip()

            # Optional: skip empty text
            if len(text) == 0:
                continue

            dataset.append({
                "utt_id": utt_id,
                "dialog_id": dialog_id,
                "video_path": video_path,
                "start": item["start"],
                "end": item["end"],
                "audio_path": audio_path,
                "text": text,
                "label": item["label"]
            })


# =============================================
# SAVE
# =============================================

df = pd.DataFrame(dataset)

print("\nTotal samples:", len(df))
print(df.head())

#  Sanity check
# print("\nLabel distribution:")
# print(df["label"].value_counts())

# print("\nMissing values:")
# print(df.isnull().sum())

# Optional global cleaning
df = df[df["text"].str.len() > 0]

df.to_csv("metadata.csv", index=False)