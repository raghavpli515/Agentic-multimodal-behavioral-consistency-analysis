import sys
sys.path.append('D:\\PROJECTS\\Multimodal_root\\MMDL_env\\Lib\\site-packages')
from moviepy import VideoFileClip
import os

def extract_audio(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path)
    print(f"Audio saved to {output_audio_path}")
