import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import segmentation_models_pytorch as smp


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


SAMPLES_FILE = "valid_samples_ALL_clean.json"

TRAIN_SPLIT_FILE = "train_indices_ALL_clean.pt"
VAL_SPLIT_FILE = "val_indices_ALL_clean.pt"
TEST_SPLIT_FILE = "test_indices_ALL_clean.pt"

RESULTS_DIR = Path("results/deeplabv3plus")
CHECKPOINT_DIR = Path("checkpoints/deeplabv3plus")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BCE_WEIGHT = 0.5
NUM_WORKERS = 9
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

    for f in sorted(files):
        name = f.lower()

        if "instancemask" in name:
            instance_file = f
        elif "mask" in name and "instancemask" not in name:
            seg_file = f
        elif name.endswith("-img.tif") or name.endswith("-img.tiff"):
            image_file = f

    return image_file, instance_file, seg_file


# =========================================================
# DATASET
# =========================================================
class CellBinDBDataset(Dataset):
    def __init__(self, root_dir, target_size=(256, 256)):
        self.root_dir = root_dir
        self.target_size = target_size

        samples_file = "valid_samples_ALL_clean.json"

        if not os.path.exists(samples_file):
            raise FileNotFoundError(
                f"{samples_file} not found. Run the clean split script first."
            )

        with open(samples_file, "r") as f:
            relative_samples = json.load(f)

        self.samples = [
            os.path.join(self.root_dir, rel_path)
            for rel_path in relative_samples
        ]

        print(f"Loaded cleaned samples: {len(self.samples)}")

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        stain_name = os.path.basename(os.path.dirname(sample_dir))

        files = [
            f for f in os.listdir(sample_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]

        image_file, _, seg_file = identify_files(files)

        if image_file is None or seg_file is None:
            raise ValueError(
                f"Could not identify image and mask in {sample_dir}. "
                f"Files found: {files}. "
                f"Chosen image: {image_file}, chosen mask: {seg_file}"
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

        mask_raw = np.array(mask_pil, dtype=np.float32)

        if img_max == 0 or img.std() < 1e-6:
            print(f"\nWARNING: blank image detected: {sample_dir}")
            print("Files:", files)
            print("Chosen image:", image_file)
            print("Chosen mask:", seg_file)
            print("Image stats:", img.min(), img.max(), img.mean(), img.std())
            print(
                "Raw mask stats:",
                mask_raw.min(),
                mask_raw.max(),
                mask_raw.mean(),
                mask_raw.std(),
            )

        if stain_name == "mIF":
            mask = (mask_raw < 128).astype(np.float32)
        else:
            mask = (mask_raw > 0).astype(np.float32)

        image_tensor = torch.from_numpy(img).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor

# =========================================================
# MODEL
# =========================================================
def create_deeplabv3plus_model():
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None,
    )
    return model


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

    model = create_deeplabv3plus_model().to(DEVICE)
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
