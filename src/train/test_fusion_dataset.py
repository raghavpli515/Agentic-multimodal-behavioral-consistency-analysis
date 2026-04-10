from fusion_dataset import FusionDataset
from torch.utils.data import DataLoader

dataset = FusionDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for frames, mfcc, labels in loader:
    print("Frames:", frames.shape)
    print("MFCC:", mfcc.shape)
    print("Labels:", labels)
    break