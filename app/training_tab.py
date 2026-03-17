# app/training_tab.py

import sys
import os
import sqlite3
import cv2
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
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import your AI models
from models import SimpleRestorationNet, SRCNN

class WoodDataset(Dataset):
    """Custom PyTorch Dataset that reads from your SQLite Database."""
    def __init__(self, db_path='data/database.db', transform=None):
        self.transform = transform
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT blur_path, clear_path FROM samples")
        self.image_pairs = cursor.fetchall()
        conn.close()

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        blur_path, clear_path = self.image_pairs[idx]
        blur_img = cv2.cvtColor(cv2.imread(blur_path), cv2.COLOR_BGR2RGB)
        clear_img = cv2.cvtColor(cv2.imread(clear_path), cv2.COLOR_BGR2RGB)
        
        if self.transform:
            blur_img = self.transform(blur_img)
            clear_img = self.transform(clear_img)
            
        return blur_img, clear_img

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
            
            # Setup Model
            if self.model_name == "Simple CNN (Custom)":
                model = SimpleRestorationNet().to(device)
                save_path = "weights.pth"
            elif self.model_name == "SRCNN (Custom)":
                model = SRCNN().to(device)
                save_path = "srcnn_weights.pth"
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
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            total_batches = len(dataloader)
            
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
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                    # Update Progress Bar (Total progress across all epochs)
                    current_step = (epoch * total_batches) + batch_idx
                    total_steps = self.epochs * total_batches
                    self.progress_signal.emit(int((current_step / total_steps) * 100))
                    
                    if batch_idx % 50 == 0:
                        self.log_signal.emit(f"Epoch [{epoch+1}/{self.epochs}] | Batch [{batch_idx}/{total_batches}] | Loss: {loss.item():.4f}")
                        
                avg_loss = running_loss / total_batches
                self.log_signal.emit(f"✅ Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")

            # Save the trained brain
            torch.save(model.state_dict(), save_path)
            self.log_signal.emit(f"🎉 Training Complete! Weights saved to '{save_path}'.")
            
            # --- EVALUATION LOOP (PSNR & SSIM) ---
            self.log_signal.emit("📊 Starting Mathematical Evaluation (PSNR/SSIM)...")
            model.eval()
            
            # Reconnect to DB to grab 50 random test images
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
                    output_tensor = model(input_tensor)
                    
                    output_img = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                    output_img = (np.clip(output_img, 0, 1) * 255).astype(np.uint8)
                    
                    total_psnr += psnr(clear_img_resized, output_img, data_range=255)
                    total_ssim += ssim(clear_img_resized, output_img, data_range=255, channel_axis=-1, win_size=3)

            avg_psnr = total_psnr / 50
            avg_ssim = total_ssim / 50
            
            self.log_signal.emit(f"📈 Final Evaluation -> PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
            self.metrics_signal.emit(avg_psnr, avg_ssim)
            
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
            "VDSR (Upcoming)", 
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
        self.batch_spinbox.setValue(8) # Safe default for 6GB VRAM
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
        
        # Initialize and start the background thread
        self.thread = AITrainingThread(model_name, epochs, batch_size)
        self.thread.log_signal.connect(self.update_console)
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.metrics_signal.connect(self.update_metrics)
        self.thread.finished_signal.connect(self.training_finished)
        self.thread.start()

    def update_console(self, text):
        self.console_output.append(text)
        # Auto-scroll to bottom
        scrollbar = self.console_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_metrics(self, psnr_val, ssim_val):
        self.psnr_label.setText(f"PSNR: {psnr_val:.2f} dB")
        self.ssim_label.setText(f"SSIM: {ssim_val:.4f}")

    def training_finished(self):
        self.btn_train.setEnabled(True)
        self.model_selector.setEnabled(True)
        QMessageBox.information(self, "Success", "Training & Evaluation Complete!\nYou can now test it in the AI Restoration Tab.")