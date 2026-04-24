import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset


# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# DATA_DIR should point to the folder that contains all stain folders
# Example:
# CellBinDB/
#   DAPI/
#   HE/
#   ...
DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.join(os.environ.get("TMPDIR", "."), "CellBinDB")
)

TRAIN_SPLIT_FILE = "train_indices_ALL.pt"
VAL_SPLIT_FILE = "val_indices_ALL.pt"
TEST_SPLIT_FILE = "test_indices_ALL.pt"

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BCE_WEIGHT = 0.5
NUM_WORKERS = 4
PIN_MEMORY = torch.cuda.is_available()
TARGET_SIZE = (512, 512)
EARLY_STOPPING_PATIENCE = 10
NUM_PREDICTION_SAMPLES = 30


# =========================================================
# HELPERS
# =========================================================
def identify_files(files):
    image_file = None
    instance_file = None
    seg_file = None

    for f in files:
        name = f.lower()

        if "instancemask" in name:
            instance_file = f
        elif "mask" in name and "instancemask" not in name:
            seg_file = f
        elif "img" in name:
            image_file = f

    return image_file, instance_file, seg_file


# =========================================================
# DATASET
# =========================================================
class CellBinDBDataset(Dataset):
    def __init__(self, root_dir, target_size=(256, 256)):
        self.root_dir = root_dir
        self.target_size = target_size
        self.samples = []

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        stain_folders = sorted(
            [
                os.path.join(self.root_dir, d)
                for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ]
        )

        if len(stain_folders) == 0:
            raise RuntimeError(f"No stain folders found in: {self.root_dir}")

        for stain_folder in stain_folders:
            sample_dirs = sorted(
                [
                    os.path.join(stain_folder, d)
                    for d in os.listdir(stain_folder)
                    if os.path.isdir(os.path.join(stain_folder, d))
                ]
            )
            self.samples.extend(sample_dirs)

        if len(self.samples) == 0:
            raise RuntimeError(f"No sample folders found in any stain folder under: {self.root_dir}")

        print(f"Found {len(stain_folders)} stain folders")
        print(f"Total samples found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        files = [
            f for f in os.listdir(sample_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]

        image_file, _, seg_file = identify_files(files)

        if image_file is None or seg_file is None:
            raise ValueError(
                f"Could not identify image and mask in {sample_dir}. Files found: {files}"
            )

        image_path = os.path.join(sample_dir, image_file)
        mask_path = os.path.join(sample_dir, seg_file)

        img_pil = Image.open(image_path)
        mask_pil = Image.open(mask_path)

        if img_pil.mode != "L":
            img_pil = img_pil.convert("L")
        if mask_pil.mode != "L":
            mask_pil = mask_pil.convert("L")

        img_pil = img_pil.resize(self.target_size, Image.BILINEAR)
        mask_pil = mask_pil.resize(self.target_size, Image.NEAREST)

        img = np.array(img_pil, dtype=np.float32)
        img_max = img.max()
        if img_max > 0:
            img = img / img_max

        mask = np.array(mask_pil, dtype=np.float32)
        mask = (mask > 0).astype(np.float32)

        image_tensor = torch.from_numpy(img).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor


# =========================================================
# MODEL
# =========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Encoder
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        # Decoder (nested skip connections)
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = ConvBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = ConvBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = ConvBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(
            torch.cat([x0_0, self.up(x1_0)], 1)
        )

        x1_1 = self.conv1_1(
            torch.cat([x1_0, self.up(x2_0)], 1)
        )

        x2_1 = self.conv2_1(
            torch.cat([x2_0, self.up(x3_0)], 1)
        )

        x3_1 = self.conv3_1(
            torch.cat([x3_0, self.up(x4_0)], 1)
        )

        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, self.up(x1_1)], 1)
        )

        x1_2 = self.conv1_2(
            torch.cat([x1_0, x1_1, self.up(x2_1)], 1)
        )

        x2_2 = self.conv2_2(
            torch.cat([x2_0, x2_1, self.up(x3_1)], 1)
        )

        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1)
        )

        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1)
        )

        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1)
        )

        return self.final(x0_4)


# =========================================================
# LOSSES / METRICS
# =========================================================
def dice_loss_from_logits(logits, target, eps=1e-8):
    probs = torch.sigmoid(logits)

    probs = probs.view(-1)
    target = target.view(-1)

    intersection = (probs * target).sum()
    dice = (2 * intersection + eps) / (probs.sum() + target.sum() + eps)
    return 1 - dice


def compute_metrics_from_logits(logits, target, threshold=0.5, eps=1e-8):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(-1)
    target = target.view(-1)

    tp = (preds * target).sum()
    fp = (preds * (1 - target)).sum()
    fn = ((1 - preds) * target).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "iou": iou.item(),
        "dice": dice.item(),
    }


# =========================================================
# TRAIN / VAL
# =========================================================
def train_one_epoch(model, loader, optimizer, bce_weight=0.5):
    model.train()
    bce_loss = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    metric_sums = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0, "dice": 0.0}
    n_batches = 0

    for images, masks in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        logits = model(images)

        loss_bce = bce_loss(logits, masks)
        loss_dice = dice_loss_from_logits(logits, masks)
        loss = bce_weight * loss_bce + (1 - bce_weight) * loss_dice

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        metrics = compute_metrics_from_logits(logits.detach(), masks)
        for key in metric_sums:
            metric_sums[key] += metrics[key]

        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    avg_metrics = {k: metric_sums[k] / max(n_batches, 1) for k in metric_sums}
    return avg_loss, avg_metrics


def validate_one_epoch(model, loader, bce_weight=0.5):
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    metric_sums = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0, "dice": 0.0}
    n_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            logits = model(images)

            loss_bce = bce_loss(logits, masks)
            loss_dice = dice_loss_from_logits(logits, masks)
            loss = bce_weight * loss_bce + (1 - bce_weight) * loss_dice

            running_loss += loss.item()

            metrics = compute_metrics_from_logits(logits, masks)
            for key in metric_sums:
                metric_sums[key] += metrics[key]

            n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    avg_metrics = {k: metric_sums[k] / max(n_batches, 1) for k in metric_sums}
    return avg_loss, avg_metrics


# =========================================================
# VISUALIZATION
# =========================================================
def save_training_plots(history, out_dir=RESULTS_DIR):
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history["train_dice"], label="Train Dice")
    plt.plot(history["val_dice"], label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Training and Validation Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "dice_curve.png")
    plt.close()


def save_predictions(model, loader, num_samples=30, out_dir=RESULTS_DIR / "predictions"):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    shown = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE, non_blocking=True)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            for i in range(images.shape[0]):
                if shown >= num_samples:
                    return

                image_np = images[i].squeeze().cpu().numpy()
                mask_np = masks[i].squeeze().cpu().numpy()
                prob_np = probs[i].squeeze().cpu().numpy()
                pred_np = preds[i].squeeze().cpu().numpy()

                plt.figure(figsize=(12, 3))

                plt.subplot(1, 4, 1)
                plt.imshow(image_np, cmap="gray", vmin=0, vmax=1)
                plt.title("Input")
                plt.axis("off")

                plt.subplot(1, 4, 2)
                plt.imshow(mask_np, cmap="gray")
                plt.title("Ground Truth")
                plt.axis("off")

                plt.subplot(1, 4, 3)
                plt.imshow(prob_np, cmap="gray", vmin=0, vmax=1)
                plt.title("Probability")
                plt.axis("off")

                plt.subplot(1, 4, 4)
                plt.imshow(pred_np, cmap="gray")
                plt.title("Prediction")
                plt.axis("off")

                plt.tight_layout()
                plt.savefig(out_dir / f"prediction_{shown}.png")
                plt.close()

                shown += 1


# =========================================================
# MAIN
# =========================================================
def main():
    print(f"Using data directory: {DATA_DIR}")

    dataset = CellBinDBDataset(root_dir=DATA_DIR, target_size=TARGET_SIZE)

    train_indices = torch.load(TRAIN_SPLIT_FILE)
    val_indices = torch.load(VAL_SPLIT_FILE)
    test_indices = torch.load(TEST_SPLIT_FILE)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(f"Train size: {len(train_set)}")
    print(f"Val size: {len(val_set)}")
    print(f"Test size: {len(test_set)}")

    model = UNetPlusPlus().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_dice = -1.0
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "train_iou": [],
        "train_dice": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_iou": [],
        "val_dice": [],
    }

    for epoch in range(NUM_EPOCHS):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, bce_weight=BCE_WEIGHT
        )
        val_loss, val_metrics = validate_one_epoch(
            model, val_loader, bce_weight=BCE_WEIGHT
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        for metric_name in ["precision", "recall", "f1", "iou", "dice"]:
            history[f"train_{metric_name}"].append(train_metrics[metric_name])
            history[f"val_{metric_name}"].append(val_metrics[metric_name])

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"Train -> Precision: {train_metrics['precision']:.4f}, "
            f"Recall: {train_metrics['recall']:.4f}, "
            f"F1: {train_metrics['f1']:.4f}, "
            f"IoU: {train_metrics['iou']:.4f}, "
            f"Dice: {train_metrics['dice']:.4f}"
        )
        print(
            f"Val   -> Precision: {val_metrics['precision']:.4f}, "
            f"Recall: {val_metrics['recall']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}, "
            f"IoU: {val_metrics['iou']:.4f}, "
            f"Dice: {val_metrics['dice']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_unet_model.pt")
            print("Best model saved.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "best_unet_model.pt", map_location=DEVICE)
    )

    test_loss, test_metrics = validate_one_epoch(
        model, test_loader, bce_weight=BCE_WEIGHT
    )

    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(
        f"Test -> Precision: {test_metrics['precision']:.4f}, "
        f"Recall: {test_metrics['recall']:.4f}, "
        f"F1: {test_metrics['f1']:.4f}, "
        f"IoU: {test_metrics['iou']:.4f}, "
        f"Dice: {test_metrics['dice']:.4f}"
    )

    summary = {
        "data_dir": DATA_DIR,
        "target_size": TARGET_SIZE,
        "num_epochs_requested": NUM_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "best_val_dice": best_val_dice,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
    }

    with open(RESULTS_DIR / "test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(RESULTS_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    save_training_plots(history)
    save_predictions(model, test_loader, num_samples=NUM_PREDICTION_SAMPLES)

    print(f"Saved best model to: {CHECKPOINT_DIR / 'best_unet_model.pt'}")
    print(f"Saved metrics to: {RESULTS_DIR / 'test_metrics.json'}")
    print(f"Saved history to: {RESULTS_DIR / 'history.json'}")
    print(f"Saved plots to: {RESULTS_DIR}")
    print(f"Saved predictions to: {RESULTS_DIR / 'predictions'}")


if __name__ == "__main__":
    main()
