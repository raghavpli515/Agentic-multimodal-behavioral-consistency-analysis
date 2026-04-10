from audio_dataset import AudioDataset
from torch.utils.data import DataLoader

dataset = AudioDataset()
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for mfcc, labels in loader:
    print("MFCC shape:", mfcc.shape)
    print("Labels:", labels)
    break