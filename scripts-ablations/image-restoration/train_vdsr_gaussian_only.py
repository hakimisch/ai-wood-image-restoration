"""
scripts-ablations/image-restoration/train_vdsr_gaussian_only.py

VDSR Ablation — Fix #6: Train VDSR on NAIVE Gaussian-only blurs to demonstrate the
optical domain gap. This is Variant A of the ablation study.

Paper claim (§1.5, §4.2):
  "Models trained on synthetic Gaussian blurs alone catastrophically fail on
   physically realistic blur, while the compound physics pipeline ensures
   robust generalization."

This script:
  1. Generates a Gaussian-only blur dataset (random sigma, no compound degradation)
     to   data/blurred_gaussian/
  2. Trains VDSR for 20 epochs with the same hyperparams as the production run
     (batch_size=8, accum_steps=2, lr=1e-4, hybrid loss, cosine annealing)
  3. Saves weights to   20e_vdsr_gaussian_only_hybrid.pth

Expected outcome:
  - On Gaussian-only test set:  PSNR ≈ 18-20 (reasonable, similar to physics)
  - On physics-pipeline test set: PSNR ≪ 18 (catastrophic failure, proves domain gap)

Usage:
  cd /mnt/c/Hakimi/Internship/GUI Image Restoration
  python -X utf8 scripts-ablations/image-restoration/train_vdsr_gaussian_only.py

Runtime: ~45 min on GTX 1660 Ti (6 GB)
"""

import os
import sys
import sqlite3
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'app'))

from models import VDSR

# ── Configuration ────────────────────────────────────────────────────────────
DB_PATH      = os.path.join(PROJECT_ROOT, 'data/database.db')
CLEAR_DIR    = os.path.join(PROJECT_ROOT, 'data')
BLUR_OUT_DIR = os.path.join(PROJECT_ROOT, 'data/blurred_gaussian')

EPOCHS       = 20
BATCH_SIZE   = 8
ACCUM_STEPS  = 2          # effective batch = 16
LR           = 1e-4
LOSS_TYPE    = "Hybrid"   # 0.8 MSE + 0.2 L1
CROP_SIZE    = 256
SAVE_NAME    = os.path.join(PROJECT_ROOT, "20e_vdsr_gaussian_only_hybrid.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {DEVICE}")

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════
# PART 1 – Generate Gaussian-Only Blur Dataset
# ═══════════════════════════════════════════════════════════════════════════

def make_gaussian_blur(img_rgb, sigma):
    """Standard Gaussian blur with kernel size derived from sigma."""
    k = max(3, int(np.ceil(6 * sigma)) | 1)
    return cv2.GaussianBlur(img_rgb, (k, k), sigmaX=sigma, sigmaY=sigma)


def generate_gaussian_blurs():
    """
    Read every clear image from DB, apply ONLY Gaussian blur (random sigma),
    and save to data/blurred_gaussian/.
    No motion, no defocus, no banding, no ISP, no vignetting.
    """
    os.makedirs(BLUR_OUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, clear_path FROM samples")
    rows = cursor.fetchall()
    conn.close()

    total = len(rows)
    print(f"\n{'='*60}")
    print(f"📦 Generating Gaussian-only blurs for {total} samples...")
    print(f"   Output: {BLUR_OUT_DIR}")
    print(f"{'='*60}")

    for i, (sample_id, clear_path) in enumerate(rows):
        img_bgr = cv2.imread(clear_path)
        if img_bgr is None:
            print(f"  ⚠️  Cannot read {clear_path}, skipping.")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ── ONLY GAUSSIAN BLUR ──────────────────────────────────────────
        sigma = random.uniform(0.5, 5.0)
        blurred_rgb = make_gaussian_blur(img_rgb, sigma)
        # ────────────────────────────────────────────────────────────────

        # Save
        filename  = os.path.basename(clear_path)
        blur_path = os.path.join(BLUR_OUT_DIR, filename).replace('\\', '/')
        blurred_bgr = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(blur_path, blurred_bgr)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] Done.")

    print(f"✅ Gaussian-only blur generation complete.\n")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 – Dataset (mirrors WoodDataset from training_tab.py)
# ═══════════════════════════════════════════════════════════════════════════

class AblationWoodDataset(Dataset):
    """
    Reads (blur, clear) pairs from data/blurred_gaussian/ and the DB.
    Same train/test split (seed 42, last 50 for test) and same
    augmentation (random crop, flip, rotate) as the production pipeline.
    """

    BLUR_DIR = BLUR_OUT_DIR

    def __init__(self, split='train'):
        self.split = split
        self.crop_size = CROP_SIZE

        # Build filename -> clear_path lookup from DB
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT clear_path FROM samples")
        clear_by_name = {os.path.basename(row[0]): row[0] for row in cursor.fetchall()}
        conn.close()

        if not os.path.isdir(self.BLUR_DIR):
            raise RuntimeError(
                f"'{self.BLUR_DIR}' not found.\n"
                "Run PART 1 (generate_gaussian_blurs) first."
            )

        self.image_pairs = []
        skipped = 0
        for fname in os.listdir(self.BLUR_DIR):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            clear_path = clear_by_name.get(fname)
            if clear_path is None:
                skipped += 1
                continue
            blur_path = os.path.join(self.BLUR_DIR, fname).replace('\\', '/')
            self.image_pairs.append((blur_path, clear_path))

        # Train/test split (seed 42, same as production)
        self.image_pairs.sort()
        rng = random.Random(42)
        rng.shuffle(self.image_pairs)

        if len(self.image_pairs) > 50:
            if self.split == 'train':
                self.image_pairs = self.image_pairs[:-50]
            elif self.split == 'test':
                self.image_pairs = self.image_pairs[-50:]
        else:
            if self.split == 'test':
                self.image_pairs = self.image_pairs[:1]
            elif self.split == 'train':
                self.image_pairs = self.image_pairs[1:]

        if not self.image_pairs:
            raise RuntimeError(f"No pairs found for split '{self.split}'.")

        msg = f"📦 Dataset: {len(self.image_pairs)} pairs (Split: {self.split.upper()})"
        if skipped and self.split == 'train':
            msg += f" ({skipped} unmatched files skipped)"
        print(f"   {msg}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        blur_path, clear_path = self.image_pairs[idx]

        blur_np  = cv2.cvtColor(cv2.imread(blur_path),  cv2.COLOR_BGR2RGB)
        clear_np = cv2.cvtColor(cv2.imread(clear_path), cv2.COLOR_BGR2RGB)

        # Same target-smoothing as production (prevents over-sharpening)
        target_sigma = random.uniform(0.2, 0.6)
        clear_np = cv2.GaussianBlur(clear_np, (3, 3), target_sigma)

        # Random crop
        h, w = blur_np.shape[:2]
        top  = random.randint(0, max(0, h - self.crop_size))
        left = random.randint(0, max(0, w - self.crop_size))
        blur_np  = blur_np[top:top + self.crop_size, left:left + self.crop_size]
        clear_np = clear_np[top:top + self.crop_size, left:left + self.crop_size]

        # Data augmentation (shared flip, rotate)
        if random.random() > 0.5:
            blur_np  = blur_np[:, ::-1].copy()
            clear_np = clear_np[:, ::-1].copy()
        if random.random() > 0.5:
            blur_np  = blur_np[::-1].copy()
            clear_np = clear_np[::-1].copy()
        if random.random() > 0.5:
            k = random.randint(1, 3)
            blur_np  = np.rot90(blur_np, k).copy()
            clear_np = np.rot90(clear_np, k).copy()

        # HWC uint8 → CHW float32 [0, 1]
        blur_t  = torch.from_numpy(blur_np.transpose(2, 0, 1)).float()  / 255.0
        clear_t = torch.from_numpy(clear_np.transpose(2, 0, 1)).float() / 255.0
        return blur_t, clear_t


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 – Training Loop (mirrors AITrainingThread)
# ═══════════════════════════════════════════════════════════════════════════

def train():
    print(f"\n{'='*60}")
    print(f"🧠 Training VDSR on GAUSSIAN-ONLY blurs")
    print(f"   Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Accum: {ACCUM_STEPS}")
    print(f"   Effective batch: {BATCH_SIZE * ACCUM_STEPS}")
    print(f"   Loss: {LOSS_TYPE} | LR: {LR}")
    print(f"   Save: {SAVE_NAME}")
    print(f"{'='*60}\n")

    # Model
    model = VDSR().to(DEVICE)

    # Dataset
    dataset = AblationWoodDataset(split='train')
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Loss
    l1_loss  = nn.L1Loss()
    mse_loss = nn.MSELoss()

    def criterion(pred, target):
        return 0.2 * l1_loss(pred, target) + 0.8 * mse_loss(pred, target)

    # Optimiser + Scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    print(f"📉 CosineAnnealingLR: {LR:.0e} → 1e-6 over {EPOCHS} epochs")

    # Training loop
    total_batches = len(dataloader)
    if total_batches == 0:
        raise RuntimeError("Dataloader is empty — no training pairs found.")
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (blur_imgs, clear_imgs) in enumerate(dataloader):
            blur_imgs  = blur_imgs.to(DEVICE)
            clear_imgs = clear_imgs.to(DEVICE)

            outputs = model(blur_imgs)
            loss    = criterion(outputs, clear_imgs)
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

            if batch_idx % 50 == 0:
                accum_flag = "⚡" if (batch_idx + 1) % ACCUM_STEPS == 0 else " "
                print(f"   Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{total_batches}] | Loss: {loss.item():.4f} {accum_flag}", end='\r')

            last_batch = batch_idx  # track for gradient flush below

        # Flush remaining gradients
        if (last_batch + 1) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        epoch_loss = running_loss / total_batches
        current_lr = scheduler.get_last_lr()[0]

        print(f"   ✅ Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | LR: {current_lr:.2e}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), SAVE_NAME)
            print(f"      🏆 New best loss! Saved to {os.path.basename(SAVE_NAME)}")

    print(f"\n🎉 Training complete! Best weights: {SAVE_NAME} (loss={best_loss:.4f})")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    generate_gaussian_blurs()
    train()
    print("\n💡 To evaluate this model against the physics-pipeline test set:")
    print(f"   python evaluate.py --weights 20e_vdsr_gaussian_only_hybrid.pth --model VDSR")
    print("   (or use the GUI Evaluation tab → Load weights)")
