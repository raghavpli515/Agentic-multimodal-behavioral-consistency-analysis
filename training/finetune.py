import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import numpy as np


from dataset_finetune import FineTuneDataset
from models import FusionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CONFIG
# =========================
BATCH_SIZE = 1   # video = heavy
LR = 1e-4
EPOCHS = 15
PATIENCE = 6

# =========================
# LOAD DATASET
# =========================
ROOT = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(ROOT, "personal_dataset_with_video.csv")
dataset = FineTuneDataset(csv_path)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

# =========================
# LOAD MODEL
# =========================
model = FusionModel().to(DEVICE)

# Load pretrained weights (exclude old classifier)
state_dict = torch.load("checkpoints/best_model.pt", map_location=DEVICE)
state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}

model.load_state_dict(state_dict, strict=False)

print(" Loaded pretrained weights")

# =========================
# FREEZE BERT ONLY
# =========================
for param in model.text_model.bert.parameters():
    param.requires_grad = False

print(" BERT frozen")

# =========================
# CLASS WEIGHTS (IMPORTANT)
# =========================
labels = [dataset.df.iloc[i]["label"] for i in range(len(dataset))]
class_counts = np.bincount(labels)

weights = 1.0 / (class_counts + 1e-6)
weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler(DEVICE)

# =========================
# EARLY STOPPING
# =========================
best_val_acc = 0
patience_counter = 0

os.makedirs("checkpoints", exist_ok=True)

# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):

    # ===== TRAIN =====
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in loop:

        audio = batch["audio"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        video = batch["video"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            outputs = model(audio, input_ids, attention_mask, video)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item())

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:

            audio = batch["audio"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            video = batch["video"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(audio, input_ids, attention_mask, video)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    print("\n==============================")
    print(f"Epoch {epoch}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # ===== SAVE BEST =====
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0

        torch.save(model.state_dict(), "checkpoints/video_personal_best_2.pt")
        print(" Saved BEST VIDEO model")

    else:
        patience_counter += 1
        print(f" No improvement ({patience_counter}/{PATIENCE})")

    # ===== EARLY STOP =====
    if patience_counter >= PATIENCE:
        print("\n Early stopping triggered")
        break