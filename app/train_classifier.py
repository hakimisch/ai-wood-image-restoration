# app/train_classifier.py
#
# Training script for the Species Classifier.
# Reads clear images from Kayu/ directory structure, trains a ResNet18
# via transfer learning, and saves weights to classifier_weights.pth.
#
# Usage:
#   python app/train_classifier.py --epochs 30 --batch_size 32 --lr 1e-4
#
# The script automatically:
#   - Reads species from the DB registry
#   - Splits data 80/20 stratified by species
#   - Applies augmentation (random crop, flip, rotate, color jitter)
#   - Logs training progress and saves best weights

import os
import sys
import argparse
import sqlite3
import random
import numpy as np
import cv2
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Add parent directory to path so we can import classifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.classifier import SpeciesClassifier, build_species_index

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KAYU_DIR = "Kayu"
DB_PATH = "data/database.db"
DEFAULT_SAVE_PATH = "classifier_weights.pth"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WoodClassificationDataset(Dataset):
    """Reads clear images from Kayu/<Species>/<Block>/clear/ and assigns labels.

    Each species folder is mapped to an integer label via the DB registry.
    """

    def __init__(self, db_path=DB_PATH, kayu_dir=KAYU_DIR, split="train",
                 val_ratio=0.2, seed=42, augment=True):
        """
        Args:
            db_path: Path to SQLite database.
            kayu_dir: Root directory containing species folders.
            split: 'train' or 'val'.
            val_ratio: Fraction of images per species to hold out for validation.
            seed: Random seed for reproducible splits.
            augment: Apply data augmentation (training only).
        """
        self.kayu_dir = kayu_dir
        self.augment = augment and (split == "train")

        # Build species index from DB
        self.idx_to_name, self.name_to_idx = build_species_index(db_path)
        self.num_species = len(self.idx_to_name)
        print(f"📋 Species registry loaded: {self.num_species} species")

        # Collect all image paths with their species labels
        all_samples = []  # list of (image_path, label_index)
        skipped_species = []

        for species_name in os.listdir(kayu_dir):
            species_path = os.path.join(kayu_dir, species_name)
            if not os.path.isdir(species_path):
                continue

            # Map species folder name to label index
            if species_name not in self.name_to_idx:
                skipped_species.append(species_name)
                continue
            label = self.name_to_idx[species_name]

            # Walk through all blocks and clear/ directories
            for block in os.listdir(species_path):
                block_path = os.path.join(species_path, block)
                if not os.path.isdir(block_path):
                    continue
                clear_dir = os.path.join(block_path, "clear")
                if not os.path.isdir(clear_dir):
                    continue

                for img_file in os.listdir(clear_dir):
                    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    img_path = os.path.join(clear_dir, img_file).replace('\\', '/')
                    all_samples.append((img_path, label))

        if skipped_species:
            print(f"⚠️  Skipped {len(skipped_species)} folders not in registry: {skipped_species}")

        if not all_samples:
            raise RuntimeError(
                f"No images found in '{kayu_dir}'. Ensure the directory exists "
                "and contains species subfolders with clear/ subdirectories."
            )

        print(f"📦 Total images found: {len(all_samples)}")

        # Stratified train/val split
        rng = random.Random(seed)
        # Group by label
        by_label = {}
        for path, label in all_samples:
            by_label.setdefault(label, []).append((path, label))

        train_samples = []
        val_samples = []
        for label, samples in by_label.items():
            rng.shuffle(samples)
            n_val = max(1, int(len(samples) * val_ratio))
            val_samples.extend(samples[:n_val])
            train_samples.extend(samples[n_val:])

        if split == "train":
            self.samples = train_samples
        else:
            self.samples = val_samples

        rng.shuffle(self.samples)
        print(f"📊 Split '{split}': {len(self.samples)} images")

        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Read image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise IOError(f"Could not read image: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to 256x256 first (most Kayu images are larger)
        img_rgb = cv2.resize(img_rgb, (256, 256))

        # Data augmentation (training only)
        if self.augment:
            # Random crop to 224x224
            h, w = img_rgb.shape[:2]
            top = random.randint(0, h - 224)
            left = random.randint(0, w - 224)
            img_rgb = img_rgb[top:top + 224, left:left + 224]

            # Random horizontal flip
            if random.random() > 0.5:
                img_rgb = img_rgb[:, ::-1].copy()

            # Random vertical flip
            if random.random() > 0.5:
                img_rgb = img_rgb[::-1].copy()

            # Random rotation (90, 180, 270)
            if random.random() > 0.5:
                k = random.randint(1, 3)
                img_rgb = np.rot90(img_rgb, k).copy()

            # Color jitter (brightness, contrast, saturation)
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)
                beta = random.randint(-15, 15)
                img_rgb = cv2.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)
        else:
            # Validation: center crop to 224x224
            h, w = img_rgb.shape[:2]
            top = (h - 224) // 2
            left = (w - 224) // 2
            img_rgb = img_rgb[top:top + 224, left:left + 224]

        # Convert to tensor and normalize
        tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        tensor = self.normalize(tensor)

        return tensor, label


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # Reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    # Datasets
    train_dataset = WoodClassificationDataset(
        db_path=DB_PATH,
        kayu_dir=KAYU_DIR,
        split="train",
        val_ratio=args.val_ratio,
        seed=42,
        augment=True,
    )
    val_dataset = WoodClassificationDataset(
        db_path=DB_PATH,
        kayu_dir=KAYU_DIR,
        split="val",
        val_ratio=args.val_ratio,
        seed=42,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Model
    model = SpeciesClassifier(
        num_species=train_dataset.num_species,
        freeze_backbone=not args.unfreeze,
    )
    model = model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    print(f"\n{'='*60}")
    print(f"Model: ResNet18 | Species: {model.num_species}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Freeze backbone: {not args.unfreeze}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"{'='*60}\n")

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 50 == 0:
                print(f"  Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"\n📊 Epoch {epoch+1}/{args.epochs} — "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            model.save_weights(args.save_path)
            print(f"🌟 New best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")

        print()

    print(f"\n{'='*60}")
    print(f"🎉 Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Weights saved to: {args.save_path}")
    print(f"{'='*60}")

    # Save training metadata to DB
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''CREATE TABLE IF NOT EXISTS classifier_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            num_species INTEGER,
            epochs INTEGER,
            batch_size INTEGER,
            learning_rate REAL,
            freeze_backbone INTEGER,
            best_val_acc REAL,
            best_epoch INTEGER,
            timestamp TEXT
        )''')
        conn.execute(
            "INSERT INTO classifier_metrics "
            "(model_name, num_species, epochs, batch_size, learning_rate, "
            " freeze_backbone, best_val_acc, best_epoch, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("ResNet18", model.num_species, args.epochs, args.batch_size,
             args.lr, int(not args.unfreeze), best_val_acc, best_epoch,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
        print("💾 Training metrics saved to database.")
    except Exception as e:
        print(f"⚠️  Could not save metrics to DB: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet18 species classifier on Kayu wood dataset."
    )
    parser.add_argument("--epochs", "-e", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch_size", "-b", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", "-l", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--save_path", "-s", type=str, default=DEFAULT_SAVE_PATH,
                        help=f"Path to save weights (default: {DEFAULT_SAVE_PATH})")
    parser.add_argument("--val_ratio", "-v", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    parser.add_argument("--unfreeze", "-u", action="store_true",
                        help="Unfreeze backbone for full fine-tuning")
    args = parser.parse_args()

    train_classifier(args)


if __name__ == "__main__":
    main()
