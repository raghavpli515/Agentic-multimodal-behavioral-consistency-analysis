import cv2

def segment_video(video_path, segment_seconds=2, fps=5):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f" Cannot open video file: {video_path}")

    frames = []
    segments = []

    video_fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the original video FPS to calculate frame intervals for segmenting. This is crucial for ensuring that we extract frames at the correct intervals based on the desired segment length and target FPS. If the video FPS is 30 and we want segments of 2 seconds at 5 FPS, we need to know the original FPS to determine how many frames to skip between each extracted frame.

    if video_fps == 0:
        raise ValueError(" FPS is 0 — invalid or unsupported video")

    frame_interval = max(1, video_fps // fps)
    segment_size = segment_seconds * fps

    count = 0

    while True:   #  safer than cap.isOpened()
        try:
            ret, frame = cap.read()
        except Exception as e:
            print(" Frame read error:", e)
            break

        if not ret:
            break

        if frame is None:
            continue  # skip corrupted frames

        if count % frame_interval == 0:
            frames.append(frame)

            if len(frames) == segment_size:
                start_time = (count - len(frames) * frame_interval) / video_fps
                end_time = count / video_fps

                segments.append({
                    "frames": frames.copy(),  # copy to avoid reference issues, frames will be cleared in next segment
                    "start": start_time,
                    "end": end_time
                })
                
                frames = []  # reset for next segment

        count += 1

    cap.release()

    #  Important safety checks
    if len(segments) == 0:
        raise ValueError(" No segments extracted — video too short or corrupted")

    print(f"[DEBUG] Segments created: {len(segments)}")
    return segments
    