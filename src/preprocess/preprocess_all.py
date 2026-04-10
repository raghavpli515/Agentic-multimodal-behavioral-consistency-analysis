# Preprocess all videos in the dataset: extract frames, audio, and transcripts.
import os
from moviepy import VideoFileClip
import whisper
import cv2

DATASET_PATH = "dataset"
PROCESSED_PATH = "processed"
FRAME_RATE = 1  # 1 frame per second

# Load Whisper once
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading Whisper model...")
whisper_model = whisper.load_model("base").to(device)
print(f"Whisper running on: {device}")

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)   # Create output folder if it doesn't exist,exist_ok=True allows it to skip if it already exists
    cap = cv2.VideoCapture(video_path)      # Open the video file for reading using OpenCV's VideoCapture class. This allows us to read frames from the video.
    fps = cap.get(cv2.CAP_PROP_FPS)         # Get the frames per second (FPS) of the video using the CAP_PROP_FPS property. This tells us how many frames are in one second of the video.
    interval = int(fps * FRAME_RATE)        # Calculate the interval between frames to save based on the desired frame rate. For example, if the video has 30 FPS and we want 1 frame per second, the interval will be 30 (i.e., save every 30th frame).

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()             # Read the next frame from the video. The ret is a boolean variable indicates whether the frame was read successfully, and the frame variable contains the actual image data of the frame.
        if not ret:
            break

        if count % interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved}.jpg")
            cv2.imwrite(frame_path, frame)  # Save the current frame as a JPEG image in the output folder. The filename is generated using the saved counter to ensure unique names for each frame.
            saved += 1

        count += 1

    cap.release()               # Release the video capture object to free up resources. This is important to avoid memory leaks and ensure that the video file is properly closed.
    print(f"Frames extracted: {saved}")

def extract_audio(video_path, output_audio_path):
    clip = VideoFileClip(video_path)        # Load the video file using MoviePy's VideoFileClip class. This allows us to access the audio track of the video.
    clip.audio.write_audiofile(output_audio_path)   # Extract the audio from the video and save it as a separate audio file (e.g., WAV format) at the specified output path. The verbose=False and logger=None arguments are used to suppress output messages during the audio extraction process.
    print("Audio extracted.")

def transcribe_audio(audio_path, output_txt_path):
    result = whisper_model.transcribe(audio_path)
    with open(output_txt_path, "w", encoding="utf-8") as f:    # Open the output text file for writing. The encoding="utf-8" argument ensures that the file is written with UTF-8 encoding, which can handle a wide range of characters and is suitable for text data.
        f.write(result["text"])
    print("Transcript generated.")

def process_video(condition, video_file):
    video_path = os.path.join(DATASET_PATH, condition, video_file)   # Construct the full path to the video file by joining the dataset path, condition folder, and video filename. This allows us to access the video file for processing.
    base_name = video_file.replace(".mp4", "")

    frames_output = os.path.join(PROCESSED_PATH, "frames", condition, base_name)
    audio_output_folder = os.path.join(PROCESSED_PATH, "audio", condition)
    transcript_output_folder = os.path.join(PROCESSED_PATH, "transcripts", condition)

    os.makedirs(audio_output_folder, exist_ok=True)
    os.makedirs(transcript_output_folder, exist_ok=True)

    audio_path = os.path.join(audio_output_folder, base_name + ".wav")
    transcript_path = os.path.join(transcript_output_folder, base_name + ".txt")

    # Skip if already processed
    if os.path.exists(transcript_path):
        print(f"Skipping (already processed): {video_file}")
        return

    print(f"\nProcessing: {video_file}")

    extract_frames(video_path, frames_output)
    extract_audio(video_path, audio_path)
    transcribe_audio(audio_path, transcript_path)

def main():
    for condition in os.listdir(DATASET_PATH):            # Iterate through each condition folder in the dataset directory. This allows us to process videos organized by different conditions (e.g., "condition1", "condition2", etc.).
        condition_path = os.path.join(DATASET_PATH, condition)

        if os.path.isdir(condition_path):                # Check if the current path is a directory (i.e., a condition folder). This ensures that we only process folders and not any files that might be present in the dataset directory.
            for video_file in os.listdir(condition_path):  # Iterate through each video file in the current condition folder. This allows us to process all videos within that specific condition.
                if video_file.endswith(".mp4"):
                    process_video(condition, video_file)

if __name__ == "__main__":
    main()