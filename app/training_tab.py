# app/training_tab.py
#
# Prerequisite: run generate_blur_dataset.py once before training.
# That script fills data/blurred/ and writes blur_path into the DB.
# Training then reads pre-generated (blur, clear) pairs — no on-the-fly
# CPU blur math, no worker pickling overhead, maximum GPU utilisation.
 
import os
import sqlite3
import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QTextEdit, QProgressBar,
                             QGroupBox, QSpinBox, QMessageBox, QLineEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
 
from models import SimpleRestorationNet, SRCNN, VDSR, SwinIR
 
 
# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
 
class WoodDataset(Dataset):
    """Reads pre-generated (blur, clear) pairs from data/blurred/.

    The DB's blur_path column is never read or written here — old
    acquisition-tab blur paths in Kayu/.../blur/ are left completely
    untouched. Pairing is done by matching bare filenames between
    data/blurred/ and the clear_path filenames stored in the DB.
    """

    BLUR_DIR = 'data/blurred'

    def __init__(self, db_path='data/database.db', transform=None, log_fn=None, split='train', crop_size=256):
        self.transform = transform
        self.split = split
        self.crop_size = crop_size

        # Build filename -> clear_path lookup from DB (read-only, no writes)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT clear_path FROM samples")
        clear_by_name = {os.path.basename(row[0]): row[0] for row in cursor.fetchall()}
        conn.close()

        if not os.path.isdir(self.BLUR_DIR):
            raise RuntimeError(
                f"'{self.BLUR_DIR}' folder not found.\n"
                "Please run  generate_blur_dataset.py  first."
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

        # Academic Constraint: Data Leakage Prevention & Deterministic Splits
        self.image_pairs.sort() # Ensure consistent order
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
            raise RuntimeError(
                f"No matching pairs found for split '{self.split}'.\n"
                "Ensure filenames in data/blurred/ match the clear image filenames."
            )

        if log_fn:
            msg = f"📦 Dataset ready: {len(self.image_pairs)} pairs (Split: {self.split.upper()})."
            if skipped and self.split == 'train':
                msg += f" ({skipped} files in data/blurred/ had no DB match — skipped.)"
            log_fn(msg)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        blur_path, clear_path = self.image_pairs[idx]

        # Read directly as NumPy uint8 — no PIL round-trip overhead
        blur_np  = cv2.cvtColor(cv2.imread(blur_path),  cv2.COLOR_BGR2RGB)
        clear_np = cv2.cvtColor(cv2.imread(clear_path), cv2.COLOR_BGR2RGB)

        if self.transform:
            # Random crop: identical window applied to both arrays
            h, w    = blur_np.shape[:2]
            top     = random.randint(0, max(0, h - self.crop_size))
            left    = random.randint(0, max(0, w - self.crop_size))
            blur_np  = blur_np[top:top + self.crop_size, left:left + self.crop_size]
            clear_np = clear_np[top:top + self.crop_size, left:left + self.crop_size]

            # Shared horizontal flip
            if random.random() > 0.5:
                blur_np  = blur_np[:, ::-1].copy()
                clear_np = clear_np[:, ::-1].copy()

            # Shared vertical flip
            if random.random() > 0.5:
                blur_np  = blur_np[::-1].copy()
                clear_np = clear_np[::-1].copy()

            if random.random() > 0.5:
                # Rotate by 90, 180, or 270 degrees
                k = random.randint(1, 3) 
                # np.rot90 works perfectly on HWC arrays
                blur_np  = np.rot90(blur_np, k).copy()
                clear_np = np.rot90(clear_np, k).copy()

        # HWC uint8 -> CHW float32 [0,1] without touching PIL
        blur_t  = torch.from_numpy(blur_np.transpose(2, 0, 1)).float()  / 255.0
        clear_t = torch.from_numpy(clear_np.transpose(2, 0, 1)).float() / 255.0
        return blur_t, clear_t
 
 
# ---------------------------------------------------------------------------
# Training thread
# ---------------------------------------------------------------------------
 
class AITrainingThread(QThread):
    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    metrics_signal  = pyqtSignal(float, float)
    finished_signal = pyqtSignal()
 
    def __init__(self, model_name, epochs, batch_size, loss_type, save_name, use_amp, accum_steps, lr):
        super().__init__()
        self.model_name   = model_name
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.loss_type    = loss_type
        self.save_name    = save_name
        self.use_amp      = use_amp
        self.accum_steps  = accum_steps
        self.lr           = lr
 
    def run(self):
        try:
            # Academic Constraint: Enforce Reproducibility Seed
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            np.random.seed(42)
            random.seed(42)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_signal.emit(f"🚀 Initializing Training on: {device}")
            if self.use_amp:
                mode_str = "AMP (Mixed Precision FP16)"
            elif self.accum_steps > 1:
                mode_str = f"FP32 + Gradient Accumulation (×{self.accum_steps} = effective batch {self.batch_size * self.accum_steps})"
            else:
                mode_str = "FP32 Baseline"
            self.log_signal.emit(f"⚡ Training Mode: {mode_str}")
 
            # ----------------------------------------------------------------
            # Model & Dynamic Crop Size Initialization
            # ----------------------------------------------------------------
            if self.model_name == "Simple CNN (Custom)":
                crop_size = 256
                model = SimpleRestorationNet().to(device)
            elif self.model_name == "SRCNN (Custom)":
                crop_size = 256
                model = SRCNN().to(device)
            elif self.model_name == "VDSR (Custom)":
                crop_size = 256
                model = VDSR().to(device)
            elif self.model_name == "SwinIR (Custom)":
                crop_size = 128
                model = SwinIR(img_size=128).to(device)
                self.log_signal.emit("⚡ SwinIR detected: Overriding crop size to 128x128 for VRAM safety.")
            else:
                self.log_signal.emit(f"❌ {self.model_name} not implemented yet.")
                self.finished_signal.emit()
                return
 
            save_path = self.save_name
 

            # ----------------------------------------------------------------
            # Dataset + DataLoader
            # ----------------------------------------------------------------
            dataset = WoodDataset(
                db_path='data/database.db',
                transform=True,
                log_fn=self.log_signal.emit,
                split='train',
                crop_size=crop_size
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )
            # ----------------------------------------------------------------
            # Loss
            # ----------------------------------------------------------------
            l1_loss  = nn.L1Loss()
            mse_loss = nn.MSELoss()
 
            def criterion(pred, target):
                if "Hybrid" in self.loss_type:
                    return 0.2 * l1_loss(pred, target) + 0.8 * mse_loss(pred, target)
                return mse_loss(pred, target)
 
            # ----------------------------------------------------------------
            # Optimiser + Scheduler
            # ----------------------------------------------------------------
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
            self.log_signal.emit(f"📉 CosineAnnealingLR: {self.lr:.0e} → 1e-6 over {self.epochs} epochs")
 
            # AMP scaler (only used when AMP mode is active)
            scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
            torch.cuda.empty_cache()

            # Gradient accumulation only applies in FP32 mode.
            effective_accum = 1 if self.use_amp else self.accum_steps
            effective_batch  = self.batch_size * effective_accum

            total_batches = len(dataloader)
            best_loss     = float('inf')

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
                model.train()
                running_loss = 0.0
                optimizer.zero_grad(set_to_none=True)

                for batch_idx, (blur_imgs, clear_imgs) in enumerate(dataloader):
                    blur_imgs  = blur_imgs.to(device)
                    clear_imgs = clear_imgs.to(device)

                    if self.use_amp:
                        # AMP path (FP16, no accumulation)
                        with torch.amp.autocast("cuda"):
                            outputs = model(blur_imgs)
                            loss    = criterion(outputs, clear_imgs)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        running_loss += loss.item()

                    else:
                        # FP32 path (with optional gradient accumulation)
                        outputs = model(blur_imgs)
                        loss    = criterion(outputs, clear_imgs) / effective_accum
                        loss.backward()
                        running_loss += loss.item() * effective_accum

                        is_step = (batch_idx + 1) % effective_accum == 0 or (batch_idx + 1) == total_batches
                        if is_step:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)

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

                scheduler.step()
                avg_loss   = running_loss / total_batches
                current_lr = scheduler.get_last_lr()[0]
                self.log_signal.emit(
                    f"✅ Epoch {epoch+1} complete — "
                    f"Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}"
                )

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), save_path)
                    self.log_signal.emit(f"🌟 New best saved (loss: {best_loss:.4f})")
 
            torch.save(model.state_dict(), f"final_{save_path}")
            self.log_signal.emit(f"🎉 Training complete. Best weights → '{save_path}'")
 
            # ----------------------------------------------------------------
            # Evaluation (PSNR / SSIM)
            # ----------------------------------------------------------------
            self.log_signal.emit("📊 Running PSNR/SSIM evaluation on 50 holdout samples...")
            model.eval()
 
            # Load the dedicated evaluation split to prevent Data Leakage
            test_dataset = WoodDataset(
                db_path='data/database.db',
                transform=False,
                log_fn=None,
                split='test'
            )
            test_pairs = test_dataset.image_pairs
 
            eval_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
            ])
 
            total_psnr = total_ssim_val = 0.0
            with torch.no_grad():
                for blur_path, clear_path in test_pairs:
                    blur_img  = cv2.cvtColor(cv2.imread(blur_path),  cv2.COLOR_BGR2RGB)
                    clear_img = cv2.cvtColor(cv2.imread(clear_path), cv2.COLOR_BGR2RGB)
 
                    clear_tensor = eval_transform(clear_img)
                    input_tensor = eval_transform(blur_img).unsqueeze(0).to(device)
 
                    clear_np = (np.clip(clear_tensor.permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)
 
                    out_tensor = torch.clamp(model(input_tensor), 0.0, 1.0)
                    out_np     = (out_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
 
                    total_psnr     += psnr(clear_np, out_np, data_range=255)
                    total_ssim_val += ssim(clear_np, out_np, data_range=255, channel_axis=-1, win_size=3)
 
            n_eval   = max(len(test_pairs), 1)   # guard against empty list
            avg_psnr = total_psnr     / n_eval
            avg_ssim = total_ssim_val / n_eval
            self.log_signal.emit(f"📈 Holdout PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
            self.metrics_signal.emit(avg_psnr, avg_ssim)
 
            # Save metrics to DB
            conn = sqlite3.connect('data/database.db')
            conn.execute('''CREATE TABLE IF NOT EXISTS model_metrics
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 model_name TEXT, epochs INTEGER, batch_size INTEGER,
                 final_loss REAL, psnr REAL, ssim REAL, timestamp TEXT,
                 pth_filename TEXT, loss_type TEXT, accum_steps INTEGER)''')
            # Add new columns gracefully if DB was created by an older version
            for col, typedef in [
                ("pth_filename", "TEXT"),
                ("loss_type",    "TEXT"),
                ("accum_steps",  "INTEGER"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE model_metrics ADD COLUMN {col} {typedef}")
                except Exception:
                    pass  # column already exists
            conn.execute(
                "INSERT INTO model_metrics "
                "(model_name, epochs, batch_size, final_loss, psnr, ssim, "
                " timestamp, pth_filename, loss_type, accum_steps) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (self.model_name, self.epochs, self.batch_size,
                 avg_loss, avg_psnr, avg_ssim,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 os.path.basename(save_path),
                 self.loss_type,
                 self.accum_steps)
            )
            conn.commit()
            conn.close()
            self.log_signal.emit("💾 Metrics saved to database.")
 
        except Exception as e:
            self.log_signal.emit(f"❌ CRITICAL ERROR: {str(e)}")
        finally:
            self.progress_signal.emit(100)
            self.finished_signal.emit()
 
 
# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_GRP  = "QGroupBox { font-weight: bold; border: 1px solid #bdc3c7; border-radius: 6px; margin-top: 8px; padding-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #2c3e50; }"
_BLUE = "QGroupBox { font-weight: bold; border: 2px solid #2980b9; border-radius: 6px; margin-top: 8px; padding-top: 6px; background: #eaf3fb; } QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #1a5276; }"
_TIP  = "font-size: 11px; color: #7f8c8d; font-style: italic;"


class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        from PyQt6.QtWidgets import QRadioButton, QButtonGroup

        layout = QVBoxLayout()
        layout.setSpacing(6)

        # Training Configuration row
        base_group = QGroupBox("Training Configuration")
        base_group.setStyleSheet(_GRP)
        base_layout = QHBoxLayout()

        base_layout.addWidget(QLabel("Model:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "Simple CNN (Custom)", "SRCNN (Custom)", "VDSR (Custom)", "SwinIR (Custom)"
        ])
        self.model_selector.setMinimumWidth(180)
        base_layout.addWidget(self.model_selector)
        base_layout.addSpacing(10)

        base_layout.addWidget(QLabel("Epochs:"))
        self.epoch_spinbox = QSpinBox()
        self.epoch_spinbox.setRange(1, 100)
        self.epoch_spinbox.setValue(30)
        self.epoch_spinbox.setFixedWidth(70)
        base_layout.addWidget(self.epoch_spinbox)
        base_layout.addSpacing(10)

        base_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(1, 64)
        self.batch_spinbox.setValue(8)
        self.batch_spinbox.setFixedWidth(70)
        base_layout.addWidget(self.batch_spinbox)
        base_layout.addSpacing(10)
        
        base_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_selector = QComboBox()
        self.lr_selector.addItems(["1e-3", "1e-4", "5e-5", "1e-5"])
        self.lr_selector.setCurrentText("1e-4")
        self.lr_selector.setFixedWidth(70)
        base_layout.addWidget(self.lr_selector)

        base_layout.addWidget(QLabel("Loss Function:"))
        self.loss_selector = QComboBox()
        self.loss_selector.addItems(["Pure MSE (Safe Baseline)", "Hybrid (80% MSE + 20% L1)"])
        self.loss_selector.setMinimumWidth(200)
        base_layout.addWidget(self.loss_selector)
        base_layout.addStretch()
        base_group.setLayout(base_layout)

        # Precision & Batch Strategy group (blue-bordered)
        prec_group = QGroupBox("Precision & Batch Strategy")
        prec_group.setStyleSheet(_BLUE)
        prec_layout = QVBoxLayout()
        prec_layout.setSpacing(4)

        radio_row = QHBoxLayout()
        self.radio_fp32  = QRadioButton("FP32 Baseline")
        self.radio_accum = QRadioButton("FP32 + Gradient Accumulation  ★ recommended for GTX 1660 Ti")
        self.radio_amp   = QRadioButton("AMP Mixed Precision  (FP16 — best on RTX 20xx/30xx/40xx)")
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

        # Output / save row
        save_group = QGroupBox("Output")
        save_group.setStyleSheet(_GRP)
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("Save Weights As:"))
        self.save_name_input = QLineEdit()
        save_layout.addWidget(self.save_name_input, stretch=1)
        self.btn_train = QPushButton("▶  START TRAINING")
        self.btn_train.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; "
            "padding: 6px 18px; border-radius: 4px;"
        )
        self.btn_train.setMinimumHeight(34)
        self.btn_train.clicked.connect(self.start_training)
        save_layout.addWidget(self.btn_train)
        save_group.setLayout(save_layout)

        # Console
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

        # Metrics
        metrics_group = QGroupBox("Section 3.6.3: Evaluation Metrics")
        metrics_group.setStyleSheet(_GRP)
        metrics_layout = QHBoxLayout()
        self.psnr_label = QLabel("PSNR: -- dB")
        self.psnr_label.setStyleSheet("font-size: 18px; color: #2980b9;")
        self.psnr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ssim_label = QLabel("SSIM: --")
        self.ssim_label.setStyleSheet("font-size: 18px; color: #8e44ad;")
        self.ssim_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metrics_layout.addWidget(self.psnr_label)
        metrics_layout.addWidget(self.ssim_label)
        metrics_group.setLayout(metrics_layout)

        layout.addWidget(base_group)
        layout.addWidget(prec_group)
        layout.addWidget(save_group)
        layout.addWidget(console_group, stretch=1)
        layout.addWidget(metrics_group)
        self.setLayout(layout)

        # Signals
        self.model_selector.currentTextChanged.connect(self.update_default_filename)
        self.loss_selector.currentTextChanged.connect(self.update_default_filename)
        self.mode_group.idToggled.connect(self.on_mode_changed)
        self.accum_spinbox.valueChanged.connect(self.on_accum_changed)
        self.batch_spinbox.valueChanged.connect(self.on_accum_changed)

        self.on_mode_changed()
        self.update_default_filename()

    def on_accum_changed(self, *_):
        """Recompute the effective-batch tip whenever batch size or steps change."""
        if self.mode_group.checkedId() == 1:
            eff = self.batch_spinbox.value() * self.accum_spinbox.value()
            self.accum_tip.setText(
                f"effective batch = {self.batch_spinbox.value()} "
                f"\u00d7 {self.accum_spinbox.value()} = {eff}"
            )
        self.update_default_filename()

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
            0: "Standard FP32. One optimizer step per batch. Good for debugging or baseline comparisons.",
            1: "FP32 + gradient accumulation: gradients are summed over N mini-batches before each "
               "optimizer step, simulating a larger effective batch without extra VRAM. "
               "Same speed as FP32 baseline, better gradient stability. Best choice for GTX 1660 Ti.",
            2: "Mixed precision (FP16 forward/backward, FP32 optimizer state). Fastest on cards with "
               "strong second-gen+ Tensor Cores (RTX 20xx/30xx/40xx). "
               "On GTX 1660 Ti the autocast + GradScaler overhead typically makes this slower than FP32.",
        }
        self.mode_desc.setText(descs.get(mode, ""))
        self.update_default_filename()

    def update_default_filename(self, *_):
        model_text = self.model_selector.currentText()
        loss_text  = self.loss_selector.currentText()
        mode       = self.mode_group.checkedId()

        if   "Simple" in model_text: model_abbr = "sCNN"
        elif "SRCNN"  in model_text: model_abbr = "srcnn"
        elif "VDSR"   in model_text: model_abbr = "vdsr"
        elif "Swin"   in model_text: model_abbr = "swinir"
        else:                        model_abbr = "model"

        loss_abbr = "mse" if ("MSE" in loss_text and "Hybrid" not in loss_text) else "hybrid"

        if   mode == 0: mode_abbr = "fp32"
        elif mode == 1: mode_abbr = f"accum{self.accum_spinbox.value()}"
        else:           mode_abbr = "amp"

        now = datetime.now()
        self.save_name_input.setText(
            f"{model_abbr}_{loss_abbr}_{mode_abbr}_{now.month}_{now.day}.pth"
        )

    def start_training(self):
        self.btn_train.setEnabled(False)
        self.model_selector.setEnabled(False)
        self.loss_selector.setEnabled(False)
        self.lr_selector.setEnabled(False)
        self.radio_fp32.setEnabled(False)
        self.radio_accum.setEnabled(False)
        self.radio_amp.setEnabled(False)
        self.accum_spinbox.setEnabled(False)
        self.save_name_input.setEnabled(False)
        self.console_output.clear()
        self.progress_bar.setValue(0)
        self.psnr_label.setText("PSNR: Calculating...")
        self.ssim_label.setText("SSIM: Calculating...")

        save_name = self.save_name_input.text().strip()
        if not save_name.endswith(".pth"):
            save_name += ".pth"

        mode    = self.mode_group.checkedId()
        use_amp = (mode == 2)
        accum   = self.accum_spinbox.value() if mode == 1 else 1
        lr_val  = float(self.lr_selector.currentText())

        self.thread = AITrainingThread(
            model_name  = self.model_selector.currentText(),
            epochs      = self.epoch_spinbox.value(),
            batch_size  = self.batch_spinbox.value(),
            loss_type   = self.loss_selector.currentText(),
            save_name   = save_name,
            use_amp     = use_amp,
            accum_steps = accum,
            lr          = lr_val
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

    def update_metrics(self, psnr_val, ssim_val):
        self.psnr_label.setText(f"PSNR: {psnr_val:.2f} dB")
        self.ssim_label.setText(f"SSIM: {ssim_val:.4f}")

    def training_finished(self):
        self.btn_train.setEnabled(True)
        self.model_selector.setEnabled(True)
        self.loss_selector.setEnabled(True)
        self.lr_selector.setEnabled(True)
        self.radio_fp32.setEnabled(True)
        self.radio_accum.setEnabled(True)
        self.radio_amp.setEnabled(True)
        self.accum_spinbox.setEnabled(True)
        self.save_name_input.setEnabled(True)
        QMessageBox.information(
            self, "Success",
            "Training & Evaluation Complete!\nYou can now test it in the AI Restoration Tab."
        )