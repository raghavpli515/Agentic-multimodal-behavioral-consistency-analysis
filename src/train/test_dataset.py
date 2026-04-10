from video_dataset import VideoFrameDataset
from torch.utils.data import DataLoader

dataset = VideoFrameDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for frames, labels in loader:
    print("Frames shape:", frames.shape)
    print("Labels:", labels)
    break