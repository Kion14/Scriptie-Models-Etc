import os
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cellpose import models, train


DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.join(os.environ.get("TMPDIR", "."), "CellBinDB")
)

SAMPLES_FILE = "valid_samples_ALL_clean.json"

TRAIN_SPLIT_FILE = "train_indices_ALL_clean.pt"
VAL_SPLIT_FILE = "val_indices_ALL_clean.pt"
TEST_SPLIT_FILE = "test_indices_ALL_clean.pt"

RESULTS_DIR = Path("results/cellpose")
CHECKPOINT_DIR = Path("checkpoints/cellpose")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (512, 512)
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4


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


def relabel_instance_mask(mask):
    unique_vals = np.unique(mask)
    unique_vals = unique_vals[unique_vals != 0]

    relabeled = np.zeros_like(mask, dtype=np.uint16)

    for new_id, old_id in enumerate(unique_vals, start=1):
        relabeled[mask == old_id] = new_id

    return relabeled


def load_image_and_instance_mask(sample_dir):
    files = [
        f for f in os.listdir(sample_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ]

    image_file, instance_file, _ = identify_files(files)

    if image_file is None or instance_file is None:
        raise ValueError(
            f"Could not identify image/instance mask in {sample_dir}. Files: {files}"
        )

    image_path = os.path.join(sample_dir, image_file)
    mask_path = os.path.join(sample_dir, instance_file)

    img_pil = Image.open(image_path).convert("L")
    mask_pil = Image.open(mask_path)

    img_pil = img_pil.resize(TARGET_SIZE, Image.BILINEAR)
    mask_pil = mask_pil.resize(TARGET_SIZE, Image.NEAREST)

    img = np.array(img_pil, dtype=np.float32)

    if img.max() > 0:
        img = img / img.max()

    mask = np.array(mask_pil, dtype=np.uint16)
    mask = relabel_instance_mask(mask)

    if len(np.unique(mask)) <= 1:
        print(f"WARNING: empty instance mask after relabeling: {sample_dir}")

    img = img[..., np.newaxis]

    return img, mask


def main():
    with open(SAMPLES_FILE, "r") as f:
        relative_samples = json.load(f)

    samples = [
        os.path.join(DATA_DIR, rel_path)
        for rel_path in relative_samples
    ]

    train_indices = torch.load(TRAIN_SPLIT_FILE)
    val_indices = torch.load(VAL_SPLIT_FILE)

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    train_images = []
    train_masks = []
    val_images = []
    val_masks = []

    print("Loading train data...")
    for sample_dir in train_samples:
        img, mask = load_image_and_instance_mask(sample_dir)
        train_images.append(img)
        train_masks.append(mask)

    print("Loading validation data...")
    for sample_dir in val_samples:
        img, mask = load_image_and_instance_mask(sample_dir)
        val_images.append(img)
        val_masks.append(mask)

    print("Training Cellpose model...")

    model = models.CellposeModel(gpu=True)

    model_path = train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=val_images,
        test_labels=val_masks,
        save_path=str(CHECKPOINT_DIR),
        n_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        model_name="cellpose_all_instance",
    )

    print(f"Saved Cellpose model to: {model_path}")


if __name__ == "__main__":
    main()
