import whisper
import os

def transcribe_audio(audio_path, output_txt_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"Transcript saved to {output_txt_path}")
