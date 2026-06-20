import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import IEMOCAPDataset
from models import FusionModel
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# CONFIG
# -------------------------
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 0  # Windows safe
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# LOAD DATA
# -------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(ROOT, "metadata.csv")

full_dataset = IEMOCAPDataset(csv_path)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# -------------------------
# MODEL
# -------------------------
model = FusionModel().to(DEVICE)

# OPTIONAL: class weights (helps imbalance)
labels = [full_dataset.df.iloc[i]["label"] for i in range(len(full_dataset))]
class_counts = np.bincount(labels)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

scaler = torch.amp.GradScaler(DEVICE)

# -------------------------
# CHECKPOINT RESUME
# -------------------------
start_epoch = 0
best_val_acc = 0.0

checkpoint_path = os.path.join(SAVE_DIR, "last.pt")
if os.path.exists(checkpoint_path):
    print(" Resuming from checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    best_val_acc = ckpt.get("best_val_acc", 0.0)

# -------------------------
# TRAIN LOOP
# -------------------------
for epoch in range(start_epoch, EPOCHS):

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
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            outputs = model(audio, input_ids, attention_mask)
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
            labels = batch["label"].to(DEVICE)

            outputs = model(audio, input_ids, attention_mask)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    print(f"\nEpoch {epoch}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # ===== SAVE BEST MODEL =====
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
        print(" Saved BEST model")

    # ===== SAVE LAST CHECKPOINT =====
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_acc": best_val_acc
    }, checkpoint_path)