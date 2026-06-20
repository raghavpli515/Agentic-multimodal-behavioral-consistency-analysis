import torch
import numpy as np
import sys
import os
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

import seaborn as sns
import matplotlib.pyplot as plt
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(ROOT_DIR)
from dataset_finetune import FineTuneDataset
from src.models import fusion_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================================
# LOAD DATASET
# ===================================

dataset = FineTuneDataset(
    "personal_dataset_with_video.csv"
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False
)

print(f"Validation samples: {len(val_ds)}")


# ===================================
# LOAD MODEL
# ===================================

model = fusion_model.FusionModel().to(DEVICE)

state_dict = torch.load(
    "checkpoints/video_personal_best.pt",
    map_location=DEVICE
)
missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False
)

print("Missing:", missing)
print("Unexpected:", unexpected)
model.load_state_dict(
    state_dict,
    strict=False
)

model.eval()

print("Model loaded")


# ===================================
# EVALUATION
# ===================================

all_preds = []
all_labels = []

with torch.no_grad():

    for batch in val_loader:

        audio = batch["audio"].permute(0,2,1).to(DEVICE)

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        video = batch["video"].to(DEVICE)

        labels = batch["label"].to(DEVICE)

        outputs = model(
            audio,
            input_ids,
            attention_mask,
            video
        )

        preds = torch.argmax(
            outputs,
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

# ===================================
# METRICS
# ===================================

accuracy = accuracy_score(
    all_labels,
    all_preds
)

precision, recall, f1, _ = \
    precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted"
    )

print("\n========== RESULTS ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        all_labels,
        all_preds
    )
)

# ===================================
# CONFUSION MATRIX
# ===================================

cm = confusion_matrix(
    all_labels,
    all_preds
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()

print(
    "Saved confusion_matrix.png"
)