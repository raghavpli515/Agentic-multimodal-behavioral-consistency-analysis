import sys
import os

# Add project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")) 
sys.path.append(ROOT_DIR)

# Now import YOUR pipeline
from src.test.run_system import run_system   # (we may adjust this)

def process_video(video_path: str):
    """
    Wrapper around your existing system
    """
    try:
        print(f"[DEBUG] Processing video: {video_path}") 

        result = run_system(video_path)

        print(f"[DEBUG] Result: {result}")

        return result

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise e