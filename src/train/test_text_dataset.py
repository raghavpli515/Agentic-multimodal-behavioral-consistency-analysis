from text_dataset import TextDataset
from torch.utils.data import DataLoader

dataset = TextDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for input_ids, attention_mask, labels in loader:
    print("Input IDs shape:", input_ids.shape)
    print("Attention mask shape:", attention_mask.shape)
    print("Labels:", labels)
    break