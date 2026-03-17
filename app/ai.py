# app/ai_tab.py

import cv2
import numpy as np
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QMessageBox, QFileDialog, QGroupBox, QComboBox)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap

# AI/Machine Learning Imports
try:
    from models import SimpleRestorationNet, SRCNN
    import torch
    import torchvision.transforms as T
    AI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    AI_AVAILABLE = False
    print("⚠️ AI modules not found. Restoration tab will be limited.")

def calculate_vol(image):
    if len(image.shape) == 2:
        return cv2.Laplacian(image, cv2.CV_64F).var()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

class AIRestorationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.ai_available = AI_AVAILABLE
        self.live_mode = True 
        
        self.transform = T.Compose([
            T.ToPILImage(), 
            T.Resize((256, 256)), 
            T.ToTensor()
        ])
        
        self.setup_ui()
        self.load_selected_model("Simple CNN (Custom)") # Load default on startup

    def load_selected_model(self, model_name):
        """Dynamically loads the chosen AI brain into memory."""
        if not self.ai_available: return
        
        try:
            self.rest_output_view.setText(f"Loading {model_name}...")
            self.rest_output_view.setStyleSheet("background: #ecf0f1; border: 3px dashed #bdc3c7; color: #f39c12;")
            
            if model_name == "Simple CNN (Custom)":
                self.model = SimpleRestorationNet()
                weight_file = "weights.pth"
            elif model_name == "SRCNN (Custom)":
                self.model = SRCNN()
                weight_file = "srcnn_weights.pth"
            else:
                self.rest_output_view.setText(f"{model_name} is not yet implemented.\nPlease select a valid model.")
                self.rest_output_view.setStyleSheet("background: #ecf0f1; border: 3px dashed #e74c3c; color: #e74c3c;")
                self.model = None
                return

            # Check for weights and load them
            if os.path.exists(weight_file):
                self.model.load_state_dict(torch.load(weight_file, map_location='cpu'))
                self.rest_output_view.setText(f"✅ {model_name} Loaded & Ready")
                self.rest_output_view.setStyleSheet("background: #ecf0f1; border: 3px dashed #27ae60; color: #27ae60;")
            else:
                self.rest_output_view.setText(f"⚠️ {weight_file} not found.\nRunning {model_name} with blank weights.")
                self.rest_output_view.setStyleSheet("background: #ecf0f1; border: 3px dashed #e67e22; color: #e67e22;")

            self.model.eval() 
            print(f"✨ Switched AI Engine to: {model_name}")
            
        except Exception as e:
            print(f"⚠️ Error loading model {model_name}: {e}")
            self.rest_output_view.setText(f"Error loading {model_name}")

    def setup_ui(self):
        main_layout = QHBoxLayout()
        
        # --- LEFT PANEL: INPUT CONTROL ---
        input_group = QGroupBox("1. Image Source (Input)")
        input_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #7f8c8d; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; }")
        left_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_upload = QPushButton("📂 Upload Test Image")
        self.btn_upload.clicked.connect(self.upload_image)
        self.btn_resume_live = QPushButton("📷 Resume Live Camera")
        self.btn_resume_live.clicked.connect(self.resume_live)
        
        btn_layout.addWidget(self.btn_upload)
        btn_layout.addWidget(self.btn_resume_live)
        
        self.rest_input_view = QLabel("Live Input")
        self.rest_input_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rest_input_view.setStyleSheet("background: black; border: 2px solid #34495e;")
        self.rest_input_view.setMinimumSize(450, 350)
        
        self.input_vol_lbl = QLabel("Input VOL: --")
        self.input_vol_lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: #e74c3c;")
        self.input_vol_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left_layout.addLayout(btn_layout)
        left_layout.addWidget(self.rest_input_view, stretch=1)
        left_layout.addWidget(self.input_vol_lbl)
        input_group.setLayout(left_layout)
        
        # --- RIGHT PANEL: AI OUTPUT ---
        output_group = QGroupBox("2. AI Restoration Module")
        output_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #7f8c8d; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; }")
        right_layout = QVBoxLayout()
        
        # Engine Selector
        engine_layout = QHBoxLayout()
        engine_layout.addWidget(QLabel("Select AI Engine:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "Simple CNN (Custom)", 
            "SRCNN (Custom)", 
            "VDSR (Upcoming)", 
            "SwinIR (Upcoming)", 
            "Real-ESRGAN (Pre-trained)"
        ])
        self.model_selector.currentTextChanged.connect(self.load_selected_model)
        engine_layout.addWidget(self.model_selector)
        
        self.rest_output_view = QLabel("AI Restored Result\n(Awaiting Inference)")
        self.rest_output_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rest_output_view.setStyleSheet("background: #ecf0f1; border: 3px dashed #bdc3c7; color: #7f8c8d;")
        self.rest_output_view.setMinimumSize(450, 350)
        
        self.output_vol_lbl = QLabel("Output VOL: --")
        self.output_vol_lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: #27ae60;")
        self.output_vol_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_restore = QPushButton("✨ RUN AI RESTORATION")
        self.btn_restore.setMinimumHeight(50)
        self.btn_restore.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; font-size: 14px;")
        self.btn_restore.clicked.connect(self.run_inference)
        
        if not self.ai_available:
            self.btn_restore.setEnabled(False)
            self.model_selector.setEnabled(False)

        right_layout.addLayout(engine_layout)
        right_layout.addWidget(self.rest_output_view, stretch=1)
        right_layout.addWidget(self.output_vol_lbl)
        right_layout.addWidget(self.btn_restore)
        output_group.setLayout(right_layout)
        
        main_layout.addWidget(input_group, stretch=1)
        main_layout.addWidget(output_group, stretch=1)
        self.setLayout(main_layout)

    def update_live_feed(self, frame):
        if self.live_mode:
            self.current_frame = frame
            pix_rest = self._convert_cv_to_qpixmap(frame, self.rest_input_view.width(), self.rest_input_view.height())
            self.rest_input_view.setPixmap(pix_rest)

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "Kayu", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            img = cv2.imread(file_name)
            if img is not None:
                self.live_mode = False 
                self.current_frame = img
                
                pix_rest = self._convert_cv_to_qpixmap(img, self.rest_input_view.width(), self.rest_input_view.height())
                self.rest_input_view.setPixmap(pix_rest)
                vol_in = calculate_vol(img)
                self.input_vol_lbl.setText(f"Input VOL: {vol_in:.2f}")
                
                self.rest_output_view.setText("Ready for Processing...")
                self.rest_output_view.setStyleSheet("background: #ecf0f1; border: 3px solid #f1c40f;")
                self.output_vol_lbl.setText("Output VOL: --")

    def resume_live(self):
        self.live_mode = True
        self.load_selected_model(self.model_selector.currentText()) # Resets text format
        self.input_vol_lbl.setText("Input VOL: -- (Live mode ignores static VOL)")
        self.output_vol_lbl.setText("Output VOL: --")

    def run_inference(self):
        if self.current_frame is None or not self.ai_available or self.model is None:
            return
        
        try:
            vol_in = calculate_vol(self.current_frame)
            self.input_vol_lbl.setText(f"Input VOL: {vol_in:.2f}")

            img_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(img_rgb).unsqueeze(0) 
            
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            
            output_img = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            output_img = (np.clip(output_img, 0, 1) * 255).astype(np.uint8)
            output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            
            vol_out = calculate_vol(output_bgr)
            self.output_vol_lbl.setText(f"Output VOL: {vol_out:.2f}")
            
            pix = self._convert_cv_to_qpixmap(output_bgr, self.rest_output_view.width(), self.rest_output_view.height())
            self.rest_output_view.setPixmap(pix)
            self.rest_output_view.setStyleSheet("border: 3px solid #27ae60;") 
            
        except Exception as e:
            QMessageBox.critical(self, "AI Error", f"Inference failed: {str(e)}")

    def _convert_cv_to_qpixmap(self, frame, label_width, label_height):
        if len(frame.shape) == 2:
            h, w = frame.shape
            q_img = QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, ch = frame.shape
            q_img = QImage(frame.data, w, h, w * ch, QImage.Format.Format_BGR888)
        return QPixmap.fromImage(q_img).scaled(QSize(label_width, label_height), Qt.AspectRatioMode.KeepAspectRatio)