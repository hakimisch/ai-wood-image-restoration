# app/classifier_training_tab.py
#
# GUI-based training for the Species Classifier (ResNet18).
# Mirrors the structure of training_tab.py but uses CrossEntropyLoss
# and validation accuracy instead of PSNR/SSIM.
#
# Reads clear images from Kayu/<Species>/<Block>/clear/, performs
# stratified 80/20 train/val split, applies augmentation, and saves
# the best weights based on validation accuracy.

import os
import sys
import sqlite3
import random
import numpy as np
import cv2
from datetime import datetime

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QTextEdit, QProgressBar,
                             QGroupBox, QSpinBox, QMessageBox, QLineEdit,
                             QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

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
DB_PATH  = "data/database.db"
DEFAULT_SAVE_PATH = "classifier_weights.pth"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WoodClassificationDataset(Dataset):
    """Reads clear images from Kayu/<Species>/<Block>/clear/ and assigns labels.

    Each species folder is mapped to an integer label via the DB registry.
    Stratified train/val split is performed per species.
    """

    def __init__(self, db_path=DB_PATH, kayu_dir=KAYU_DIR, split="train",
                 val_ratio=0.2, seed=42, augment=True, log_fn=None):
        """
        Args:
            db_path: Path to SQLite database.
            kayu_dir: Root directory containing species folders.
            split: 'train' or 'val'.
            val_ratio: Fraction of images per species to hold out for validation.
            seed: Random seed for reproducible splits.
            augment: Apply data augmentation (training only).
            log_fn: Optional callback for log messages.
        """
        self.kayu_dir = kayu_dir
        self.augment = augment and (split == "train")
        self._log = log_fn or (lambda x: None)

        # Build species index from DB
        self.idx_to_name, self.name_to_idx = build_species_index(db_path)
        self.num_species = len(self.idx_to_name)
        self._log(f"📋 Species registry loaded: {self.num_species} species")

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
            self._log(f"⚠️  Skipped {len(skipped_species)} folders not in registry: {skipped_species}")

        if not all_samples:
            raise RuntimeError(
                f"No images found in '{kayu_dir}'. Ensure the directory exists "
                "and contains species subfolders with clear/ subdirectories."
            )

        self._log(f"📦 Total images found: {len(all_samples)}")

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
        self._log(f"📊 Split '{split}': {len(self.samples)} images")

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
# Training thread
# ---------------------------------------------------------------------------

class ClassifierTrainingThread(QThread):
    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    metrics_signal  = pyqtSignal(float, float)  # train_acc, val_acc
    finished_signal = pyqtSignal()

    def __init__(self, epochs, batch_size, lr, save_name, use_amp,
                 accum_steps, freeze_backbone):
        super().__init__()
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.lr             = lr
        self.save_name      = save_name
        self.use_amp        = use_amp
        self.accum_steps    = accum_steps
        self.freeze_backbone = freeze_backbone

    def run(self):
        try:
            # Academic Constraint: Enforce Reproducibility Seed
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            np.random.seed(42)
            random.seed(42)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_signal.emit(f"🚀 Initializing Classifier Training on: {device}")

            if self.use_amp:
                mode_str = "AMP (Mixed Precision FP16)"
            elif self.accum_steps > 1:
                mode_str = f"FP32 + Gradient Accumulation (\u00d7{self.accum_steps} = effective batch {self.batch_size * self.accum_steps})"
            else:
                mode_str = "FP32 Baseline"
            self.log_signal.emit(f"⚡ Training Mode: {mode_str}")
            self.log_signal.emit(f"🔒 Freeze Backbone: {self.freeze_backbone}")

            # ----------------------------------------------------------------
            # Datasets + DataLoaders
            # ----------------------------------------------------------------
            self.log_signal.emit("📂 Loading dataset from Kayu/ directory...")

            train_dataset = WoodClassificationDataset(
                db_path=DB_PATH,
                kayu_dir=KAYU_DIR,
                split="train",
                val_ratio=0.2,
                seed=42,
                augment=True,
                log_fn=self.log_signal.emit,
            )
            val_dataset = WoodClassificationDataset(
                db_path=DB_PATH,
                kayu_dir=KAYU_DIR,
                split="val",
                val_ratio=0.2,
                seed=42,
                augment=False,
                log_fn=self.log_signal.emit,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            num_species = train_dataset.num_species
            self.log_signal.emit(f"🧠 Species: {num_species} | "
                                 f"Train samples: {len(train_dataset)} | "
                                 f"Val samples: {len(val_dataset)}")

            # ----------------------------------------------------------------
            # Model
            # ----------------------------------------------------------------
            model = SpeciesClassifier(
                num_species=num_species,
                freeze_backbone=self.freeze_backbone,
            )
            model = model.to(device)

            # ----------------------------------------------------------------
            # Loss, Optimizer, Scheduler
            # ----------------------------------------------------------------
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.lr,
                weight_decay=1e-4,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=1e-6
            )
            self.log_signal.emit(
                f"📉 CosineAnnealingLR: {self.lr:.0e} \u2192 1e-6 over {self.epochs} epochs"
            )

            # AMP scaler (only used when AMP mode is active)
            scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
            torch.cuda.empty_cache()

            # Gradient accumulation only applies in FP32 mode.
            effective_accum = 1 if self.use_amp else self.accum_steps
            effective_batch = self.batch_size * effective_accum

            total_batches = len(train_loader)
            best_val_acc  = 0.0
            best_epoch    = 0

            self.log_signal.emit(
                f"🧠 Starting Training: {self.epochs} epochs | "
                f"Physical batch: {self.batch_size} | "
                f"Effective batch: {effective_batch} | "
                f"Batches/epoch: {total_batches}"
            )

            # ----------------------------------------------------------------
            # Training loop
            # ----------------------------------------------------------------
            for epoch in range(self.epochs):
                # --- Training ---
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                optimizer.zero_grad(set_to_none=True)

                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)

                    if self.use_amp:
                        # AMP path (FP16, no accumulation)
                        with torch.amp.autocast("cuda"):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        running_loss += loss.item()

                    else:
                        # FP32 path (with optional gradient accumulation)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels) / effective_accum
                        loss.backward()
                        running_loss += loss.item() * effective_accum

                        is_step = ((batch_idx + 1) % effective_accum == 0 or
                                   (batch_idx + 1) == total_batches)
                        if is_step:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)

                    # Track accuracy
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    # Progress
                    current_step = epoch * total_batches + batch_idx
                    total_steps  = self.epochs * total_batches
                    self.progress_signal.emit(int(current_step / total_steps * 100))

                    if batch_idx % 50 == 0:
                        display_loss = loss.item() if self.use_amp else loss.item() * effective_accum
                        self.log_signal.emit(
                            f"Epoch [{epoch+1}/{self.epochs}] | "
                            f"Batch [{batch_idx}/{total_batches}] | "
                            f"Loss: {display_loss:.4f}"
                        )

                train_loss = running_loss / total_batches
                train_acc  = 100.0 * correct / total

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

                self.log_signal.emit(
                    f"📊 Epoch {epoch+1}/{self.epochs} \u2014 "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                    f"LR: {current_lr:.6f}"
                )

                # Save best model based on validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    model.save_weights(self.save_name)
                    self.log_signal.emit(
                        f"🌟 New best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})"
                    )

            # ----------------------------------------------------------------
            # Finalise
            # ----------------------------------------------------------------
            self.log_signal.emit(
                f"🎉 Training complete! Best val acc: {best_val_acc:.2f}% (epoch {best_epoch})"
            )
            self.log_signal.emit(f"💾 Best weights saved to: {self.save_name}")
            self.metrics_signal.emit(train_acc, val_acc)

            # ----------------------------------------------------------------
            # Save training metadata to DB
            # ----------------------------------------------------------------
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
                    ("ResNet18", num_species, self.epochs, self.batch_size,
                     self.lr, int(self.freeze_backbone), best_val_acc, best_epoch,
                     datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                conn.commit()
                conn.close()
                self.log_signal.emit("💾 Training metrics saved to database.")
            except Exception as e:
                self.log_signal.emit(f"⚠️  Could not save metrics to DB: {e}")

        except Exception as e:
            self.log_signal.emit(f"❌ CRITICAL ERROR: {str(e)}")
            import traceback
            self.log_signal.emit(traceback.format_exc())
        finally:
            self.progress_signal.emit(100)
            self.finished_signal.emit()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_GRP  = ("QGroupBox { font-weight: bold; border: 1px solid #bdc3c7; "
         "border-radius: 6px; margin-top: 8px; padding-top: 6px; } "
         "QGroupBox::title { subcontrol-origin: margin; left: 10px; "
         "color: #2c3e50; }")
_BLUE = ("QGroupBox { font-weight: bold; border: 2px solid #2980b9; "
         "border-radius: 6px; margin-top: 8px; padding-top: 6px; "
         "background: #eaf3fb; } "
         "QGroupBox::title { subcontrol-origin: margin; left: 10px; "
         "color: #1a5276; }")
_TIP  = "font-size: 11px; color: #7f8c8d; font-style: italic;"


class ClassifierTrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        from PyQt6.QtWidgets import QRadioButton, QButtonGroup

        layout = QVBoxLayout()
        layout.setSpacing(6)

        # =====================================================================
        # Training Configuration row
        # =====================================================================
        base_group = QGroupBox("Training Configuration")
        base_group.setStyleSheet(_GRP)
        base_layout = QHBoxLayout()

        # Model label (fixed: ResNet18)
        base_layout.addWidget(QLabel("Model:"))
        model_label = QLabel("ResNet18 (Pretrained)")
        model_label.setStyleSheet("font-weight: bold; color: #2980b9;")
        base_layout.addWidget(model_label)
        base_layout.addSpacing(10)

        base_layout.addWidget(QLabel("Epochs:"))
        self.epoch_spinbox = QSpinBox()
        self.epoch_spinbox.setRange(1, 200)
        self.epoch_spinbox.setValue(30)
        self.epoch_spinbox.setFixedWidth(70)
        base_layout.addWidget(self.epoch_spinbox)
        base_layout.addSpacing(10)

        base_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(1, 128)
        self.batch_spinbox.setValue(32)
        self.batch_spinbox.setFixedWidth(70)
        base_layout.addWidget(self.batch_spinbox)
        base_layout.addSpacing(10)

        base_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_selector = QComboBox()
        self.lr_selector.addItems(["1e-3", "1e-4", "5e-5", "1e-5"])
        self.lr_selector.setCurrentText("1e-4")
        self.lr_selector.setFixedWidth(70)
        base_layout.addWidget(self.lr_selector)

        # Freeze backbone checkbox
        base_layout.addSpacing(10)
        self.freeze_checkbox = QCheckBox("Freeze Backbone")
        self.freeze_checkbox.setChecked(True)
        self.freeze_checkbox.setToolTip(
            "When checked, only the custom classifier head is trained.\n"
            "Uncheck to fine-tune the entire ResNet18 (requires more VRAM)."
        )
        base_layout.addWidget(self.freeze_checkbox)

        base_layout.addStretch()
        base_group.setLayout(base_layout)

        # =====================================================================
        # Precision & Batch Strategy group (blue-bordered)
        # =====================================================================
        prec_group = QGroupBox("Precision & Batch Strategy")
        prec_group.setStyleSheet(_BLUE)
        prec_layout = QVBoxLayout()
        prec_layout.setSpacing(4)

        radio_row = QHBoxLayout()
        self.radio_fp32  = QRadioButton("FP32 Baseline")
        self.radio_accum = QRadioButton(
            "FP32 + Gradient Accumulation  \u2605 recommended for GTX 1660 Ti"
        )
        self.radio_amp   = QRadioButton(
            "AMP Mixed Precision  (FP16 \u2014 best on RTX 20xx/30xx/40xx)"
        )
        self.radio_accum.setChecked(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.radio_fp32,  0)
        self.mode_group.addButton(self.radio_accum, 1)
        self.mode_group.addButton(self.radio_amp,   2)

        radio_row.addWidget(self.radio_fp32)
        radio_row.addSpacing(16)
        radio_row.addWidget(self.radio_accum)
        radio_row.addSpacing(16)
        radio_row.addWidget(self.radio_amp)
        radio_row.addStretch()

        accum_row = QHBoxLayout()
        self.accum_label   = QLabel("Accumulation Steps:")
        self.accum_spinbox = QSpinBox()
        self.accum_spinbox.setRange(2, 8)
        self.accum_spinbox.setValue(2)
        self.accum_spinbox.setFixedWidth(60)
        self.accum_tip = QLabel()
        self.accum_tip.setStyleSheet(_TIP)
        accum_row.addSpacing(20)
        accum_row.addWidget(self.accum_label)
        accum_row.addWidget(self.accum_spinbox)
        accum_row.addWidget(self.accum_tip)
        accum_row.addStretch()

        self.mode_desc = QLabel()
        self.mode_desc.setStyleSheet(_TIP)
        self.mode_desc.setWordWrap(True)

        prec_layout.addLayout(radio_row)
        prec_layout.addLayout(accum_row)
        prec_layout.addWidget(self.mode_desc)
        prec_group.setLayout(prec_layout)

        # =====================================================================
        # Output / save row
        # =====================================================================
        save_group = QGroupBox("Output")
        save_group.setStyleSheet(_GRP)
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("Save Weights As:"))
        self.save_name_input = QLineEdit()
        self.save_name_input.setText(DEFAULT_SAVE_PATH)
        save_layout.addWidget(self.save_name_input, stretch=1)
        self.btn_train = QPushButton("\u25B6  START TRAINING")
        self.btn_train.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; "
            "padding: 6px 18px; border-radius: 4px;"
        )
        self.btn_train.setMinimumHeight(34)
        self.btn_train.clicked.connect(self.start_training)
        save_layout.addWidget(self.btn_train)
        save_group.setLayout(save_layout)

        # =====================================================================
        # Console
        # =====================================================================
        console_group = QGroupBox("Live Training Console")
        console_group.setStyleSheet(_GRP)
        console_layout = QVBoxLayout()
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet(
            "background-color: #2c3e50; color: #ecf0f1; "
            "font-family: Consolas; font-size: 12px; font-weight: normal;"
        )
        console_layout.addWidget(self.console_output)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        console_layout.addWidget(self.progress_bar)
        console_group.setLayout(console_layout)

        # =====================================================================
        # Metrics
        # =====================================================================
        metrics_group = QGroupBox("Classification Metrics")
        metrics_group.setStyleSheet(_GRP)
        metrics_layout = QHBoxLayout()
        self.train_acc_label = QLabel("Train Accuracy: --")
        self.train_acc_label.setStyleSheet("font-size: 18px; color: #2980b9;")
        self.train_acc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.val_acc_label = QLabel("Val Accuracy: --")
        self.val_acc_label.setStyleSheet("font-size: 18px; color: #8e44ad;")
        self.val_acc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metrics_layout.addWidget(self.train_acc_label)
        metrics_layout.addWidget(self.val_acc_label)
        metrics_group.setLayout(metrics_layout)

        # =====================================================================
        # Assemble
        # =====================================================================
        layout.addWidget(base_group)
        layout.addWidget(prec_group)
        layout.addWidget(save_group)
        layout.addWidget(console_group, stretch=1)
        layout.addWidget(metrics_group)
        self.setLayout(layout)

        # Signals
        self.mode_group.idToggled.connect(self.on_mode_changed)
        self.accum_spinbox.valueChanged.connect(self.on_accum_changed)
        self.batch_spinbox.valueChanged.connect(self.on_accum_changed)

        self.on_mode_changed()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def on_accum_changed(self, *_):
        """Recompute the effective-batch tip whenever batch size or steps change."""
        if self.mode_group.checkedId() == 1:
            eff = self.batch_spinbox.value() * self.accum_spinbox.value()
            self.accum_tip.setText(
                f"effective batch = {self.batch_spinbox.value()} "
                f"\u00d7 {self.accum_spinbox.value()} = {eff}"
            )

    def on_mode_changed(self, *_):
        mode       = self.mode_group.checkedId()
        show_accum = (mode == 1)
        self.accum_label.setVisible(show_accum)
        self.accum_spinbox.setVisible(show_accum)
        self.accum_tip.setVisible(show_accum)
        if show_accum:
            eff = self.batch_spinbox.value() * self.accum_spinbox.value()
            self.accum_tip.setText(
                f"effective batch = {self.batch_spinbox.value()} "
                f"\u00d7 {self.accum_spinbox.value()} = {eff}"
            )
        descs = {
            0: ("Standard FP32. One optimizer step per batch. "
                "Good for debugging or baseline comparisons."),
            1: ("FP32 + gradient accumulation: gradients are summed over N "
                "mini-batches before each optimizer step, simulating a larger "
                "effective batch without extra VRAM. "
                "Same speed as FP32 baseline, better gradient stability. "
                "Best choice for GTX 1660 Ti."),
            2: ("Mixed precision (FP16 forward/backward, FP32 optimizer state). "
                "Fastest on cards with strong second-gen+ Tensor Cores "
                "(RTX 20xx/30xx/40xx). "
                "On GTX 1660 Ti the autocast + GradScaler overhead typically "
                "makes this slower than FP32."),
        }
        self.mode_desc.setText(descs.get(mode, ""))

    # ------------------------------------------------------------------
    # Training lifecycle
    # ------------------------------------------------------------------

    def start_training(self):
        self.btn_train.setEnabled(False)
        self.epoch_spinbox.setEnabled(False)
        self.batch_spinbox.setEnabled(False)
        self.lr_selector.setEnabled(False)
        self.freeze_checkbox.setEnabled(False)
        self.radio_fp32.setEnabled(False)
        self.radio_accum.setEnabled(False)
        self.radio_amp.setEnabled(False)
        self.accum_spinbox.setEnabled(False)
        self.save_name_input.setEnabled(False)
        self.console_output.clear()
        self.progress_bar.setValue(0)
        self.train_acc_label.setText("Train Accuracy: Training...")
        self.val_acc_label.setText("Val Accuracy: Training...")

        save_name = self.save_name_input.text().strip()
        if not save_name.endswith(".pth"):
            save_name += ".pth"

        mode    = self.mode_group.checkedId()
        use_amp = (mode == 2)
        accum   = self.accum_spinbox.value() if mode == 1 else 1
        lr_val  = float(self.lr_selector.currentText())

        self.thread = ClassifierTrainingThread(
            epochs          = self.epoch_spinbox.value(),
            batch_size      = self.batch_spinbox.value(),
            lr              = lr_val,
            save_name       = save_name,
            use_amp         = use_amp,
            accum_steps     = accum,
            freeze_backbone = self.freeze_checkbox.isChecked(),
        )
        self.thread.log_signal.connect(self.update_console)
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.metrics_signal.connect(self.update_metrics)
        self.thread.finished_signal.connect(self.training_finished)
        self.thread.start()

    def update_console(self, text):
        self.console_output.append(text)
        self.console_output.verticalScrollBar().setValue(
            self.console_output.verticalScrollBar().maximum()
        )

    def update_metrics(self, train_acc, val_acc):
        self.train_acc_label.setText(f"Train Accuracy: {train_acc:.2f}%")
        self.val_acc_label.setText(f"Val Accuracy: {val_acc:.2f}%")

    def training_finished(self):
        self.btn_train.setEnabled(True)
        self.epoch_spinbox.setEnabled(True)
        self.batch_spinbox.setEnabled(True)
        self.lr_selector.setEnabled(True)
        self.freeze_checkbox.setEnabled(True)
        self.radio_fp32.setEnabled(True)
        self.radio_accum.setEnabled(True)
        self.radio_amp.setEnabled(True)
        self.accum_spinbox.setEnabled(True)
        self.save_name_input.setEnabled(True)
        QMessageBox.information(
            self, "Success",
            "Classifier Training Complete!\n\n"
            "The best weights (by validation accuracy) have been saved.\n"
            "You can now test the classifier in the Species Recognition Tab."
        )
