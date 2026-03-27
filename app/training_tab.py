# app/training_tab.py

import sys
import os
import sqlite3
import cv2
import random
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QComboBox, QTextEdit, QProgressBar, 
                             QGroupBox, QSpinBox, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import your AI models
from models import SimpleRestorationNet, SRCNN, VDSR

class WoodDataset(Dataset):
    """Custom PyTorch Dataset that generates dynamic optical blurs."""
    def __init__(self, db_path='data/database.db', transform=None):
        self.transform = transform
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT clear_path FROM samples")
        self.image_paths = cursor.fetchall()
        conn.close()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        clear_path = self.image_paths[idx][0]
        
        clear_img_bgr = cv2.imread(clear_path)
        clear_img_rgb = cv2.cvtColor(clear_img_bgr, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            clear_tensor = self.transform(clear_img_rgb)
        else:
            clear_tensor = T.ToTensor()(clear_img_rgb)
            
        # The AI learns to fix everything from tiny focal issues (3x3) 
        # to massive optical blurs (21x21).
        kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21] 
        chosen_kernel = random.choice(kernel_sizes)
        
        blur_tensor = TF.gaussian_blur(clear_tensor, kernel_size=[chosen_kernel, chosen_kernel])

        return blur_tensor, clear_tensor


class AITrainingThread(QThread):
    """Background thread to handle heavy PyTorch training without freezing the GUI."""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    metrics_signal = pyqtSignal(float, float) # PSNR, SSIM
    finished_signal = pyqtSignal()

    def __init__(self, model_name, epochs, batch_size):
        super().__init__()
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_signal.emit(f"🚀 Initializing Training on: {device}")
            
            if self.model_name == "Simple CNN (Custom)":
                model = SimpleRestorationNet().to(device)
                save_path = "weights.pth"
            elif self.model_name == "SRCNN (Custom)":
                model = SRCNN().to(device)
                save_path = "srcnn_weights.pth"
            elif self.model_name == "VDSR (Custom)": 
                model = VDSR().to(device)
                save_path = "vdsr_weights.pth"
            else:
                self.log_signal.emit(f"❌ {self.model_name} not implemented yet.")
                self.finished_signal.emit()
                return

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            
            self.log_signal.emit("📦 Loading dataset from SQLite...")
            dataset = WoodDataset(db_path='data/database.db', transform=transform)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            l1_loss = nn.L1Loss()
            mse_loss = nn.MSELoss()

            def tv_loss(img):
                return torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
                       torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

            # --- UPDATED: Balanced Loss Function ---
            # 50% L1 (Sharpness) + 40% MSE (Smoothness/Safety) + 10% TV (Artifact Reduction)
            def criterion(pred, target):
                l1 = l1_loss(pred, target)
                mse = mse_loss(pred, target)
                tv = tv_loss(pred)
                return 0.5 * l1 + 0.4 * mse + 0.0001 * tv

            learning_rate = 0.0001
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            total_batches = len(dataloader)
            
            # --- CRITICAL FIX: Initialize best_loss before training loop ---
            best_loss = float('inf')
            
            self.log_signal.emit(f"🧠 Starting Training: {self.epochs} Epochs | Batch Size: {self.batch_size}")
            
            # --- TRAINING LOOP ---
            for epoch in range(self.epochs):
                model.train()
                running_loss = 0.0
    
                for batch_idx, (blur_imgs, clear_imgs) in enumerate(dataloader):
                    blur_imgs, clear_imgs = blur_imgs.to(device), clear_imgs.to(device)
        
                    optimizer.zero_grad()

                    outputs = model(blur_imgs)

                    loss = criterion(outputs, clear_imgs)
                    loss.backward()
                    
                    # --- NEW: Gradient Clipping ---
                    # Prevents the weights from exploding and causing massive VOL spikes at high epochs
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
        
                    running_loss += loss.item()
        
                    current_step = (epoch * total_batches) + batch_idx
                    total_steps = self.epochs * total_batches
                    self.progress_signal.emit(int((current_step / total_steps) * 100))
        
                    if batch_idx % 50 == 0:
                        self.log_signal.emit(
                            f"Epoch [{epoch+1}/{self.epochs}] | "
                            f"Batch [{batch_idx}/{total_batches}] | "
                            f"Loss: {loss.item():.4f}"
                        )
            
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
    
                avg_loss = running_loss / total_batches
                self.log_signal.emit(
                    f"✅ Epoch {epoch+1} Complete. "
                    f"Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}"
                )

                # --- RESTORED: Early Stopping / Best Weight Saver ---
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), save_path)
                    self.log_signal.emit(f"🌟 New best model saved! (Loss: {best_loss:.4f})")

            # Final save backup just in case
            torch.save(model.state_dict(), f"final_{save_path}")
            self.log_signal.emit(f"🎉 Training Complete! Best weights saved to '{save_path}'.")
            
            # --- EVALUATION LOOP (PSNR & SSIM) ---
            self.log_signal.emit("📊 Starting Mathematical Evaluation (PSNR/SSIM)...")
            model.eval()
            
            conn = sqlite3.connect('data/database.db')
            cursor = conn.cursor()
            cursor.execute("SELECT blur_path, clear_path FROM samples ORDER BY RANDOM() LIMIT 50")
            test_pairs = cursor.fetchall()
            conn.close()

            total_psnr, total_ssim = 0.0, 0.0
            with torch.no_grad():
                for blur_path, clear_path in test_pairs:
                    blur_img = cv2.cvtColor(cv2.imread(blur_path), cv2.COLOR_BGR2RGB)
                    clear_img = cv2.cvtColor(cv2.imread(clear_path), cv2.COLOR_BGR2RGB)
                    clear_img_resized = cv2.resize(clear_img, (256, 256))
        
                    input_tensor = transform(blur_img).unsqueeze(0).to(device)

                    output_tensor = torch.clamp(model(input_tensor), 0.0, 1.0)

                    output_img = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                    output_img = (np.clip(output_img, 0, 1) * 255).astype(np.uint8)
        
                    total_psnr += psnr(clear_img_resized, output_img, data_range=255)
                    total_ssim += ssim(
                        clear_img_resized,
                        output_img,
                        data_range=255,
                        channel_axis=-1,
                        win_size=3
                    )

            avg_psnr = total_psnr / 50
            avg_ssim = total_ssim / 50
            
            self.log_signal.emit(f"📈 Final Evaluation -> PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
            self.metrics_signal.emit(avg_psnr, avg_ssim)

            self.log_signal.emit("💾 Saving metrics to database...")
            conn = sqlite3.connect('data/database.db')
            
            conn.execute('''CREATE TABLE IF NOT EXISTS model_metrics
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             model_name TEXT,
                             epochs INTEGER,
                             batch_size INTEGER,
                             final_loss REAL,
                             psnr REAL,
                             ssim REAL,
                             timestamp TEXT)''')
                             
            conn.execute('''INSERT INTO model_metrics 
                            (model_name, epochs, batch_size, final_loss, psnr, ssim, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?)''',
                         (self.model_name, self.epochs, self.batch_size, avg_loss, avg_psnr, avg_ssim, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()
            self.log_signal.emit("✅ Metrics saved successfully!")
            
        except Exception as e:
            self.log_signal.emit(f"❌ CRITICAL ERROR: {str(e)}")
            
        finally:
            self.progress_signal.emit(100)
            self.finished_signal.emit()


class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # --- TOP: Controls ---
        control_group = QGroupBox("Training Configuration")
        control_group.setStyleSheet("font-weight: bold;")
        control_layout = QHBoxLayout()
        
        control_layout.addWidget(QLabel("Select Model:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "Simple CNN (Custom)", 
            "SRCNN (Custom)", 
            "VDSR (Custom)", 
            "SwinIR (Upcoming)"
        ])
        control_layout.addWidget(self.model_selector)
        
        control_layout.addWidget(QLabel("Epochs:"))
        self.epoch_spinbox = QSpinBox()
        self.epoch_spinbox.setRange(1, 100)
        self.epoch_spinbox.setValue(10)
        control_layout.addWidget(self.epoch_spinbox)
        
        control_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(1, 64)
        self.batch_spinbox.setValue(8) 
        control_layout.addWidget(self.batch_spinbox)
        
        self.btn_train = QPushButton("▶ START TRAINING")
        self.btn_train.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 5px;")
        self.btn_train.clicked.connect(self.start_training)
        control_layout.addWidget(self.btn_train)
        
        control_group.setLayout(control_layout)
        
        # --- MIDDLE: Live Console ---
        console_group = QGroupBox("Live Training Console")
        console_group.setStyleSheet("font-weight: bold;")
        console_layout = QVBoxLayout()
        
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("background-color: #2c3e50; color: #ecf0f1; font-family: Consolas; font-weight: normal;")
        console_layout.addWidget(self.console_output)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        console_layout.addWidget(self.progress_bar)
        
        console_group.setLayout(console_layout)
        
        # --- BOTTOM: Evaluation Metrics ---
        metrics_group = QGroupBox("Section 3.6.3: Evaluation Metrics")
        metrics_group.setStyleSheet("font-weight: bold;")
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
        
        # Add everything to main layout
        layout.addWidget(control_group)
        layout.addWidget(console_group, stretch=1)
        layout.addWidget(metrics_group)
        self.setLayout(layout)

    def start_training(self):
        """Disables buttons and starts the PyTorch thread."""
        self.btn_train.setEnabled(False)
        self.model_selector.setEnabled(False)
        self.console_output.clear()
        self.progress_bar.setValue(0)
        self.psnr_label.setText("PSNR: Calculating...")
        self.ssim_label.setText("SSIM: Calculating...")
        
        model_name = self.model_selector.currentText()
        epochs = self.epoch_spinbox.value()
        batch_size = self.batch_spinbox.value()
        
        self.thread = AITrainingThread(model_name, epochs, batch_size)
        self.thread.log_signal.connect(self.update_console)
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.metrics_signal.connect(self.update_metrics)
        self.thread.finished_signal.connect(self.training_finished)
        self.thread.start()

    def update_console(self, text):
        self.console_output.append(text)
        scrollbar = self.console_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_metrics(self, psnr_val, ssim_val):
        self.psnr_label.setText(f"PSNR: {psnr_val:.2f} dB")
        self.ssim_label.setText(f"SSIM: {ssim_val:.4f}")

    def training_finished(self):
        self.btn_train.setEnabled(True)
        self.model_selector.setEnabled(True)
        QMessageBox.information(self, "Success", "Training & Evaluation Complete!\nYou can now test it in the AI Restoration Tab.")