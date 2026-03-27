# app/ai_tab.py

import cv2
import numpy as np
import os
import math
import time # <-- NEW IMPORT FOR LIVE VOL THROTTLING
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QMessageBox, QFileDialog, QGroupBox, 
                             QComboBox, QRadioButton, QCheckBox)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap

# AI/Machine Learning Imports
try:
    from models import SimpleRestorationNet, SRCNN, VDSR, SwinIR
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
        
        # --- NEW: Live VOL Management ---
        self.last_vol_time = 0
        self.vol_update_interval = 0.05 # Throttle math to prevent GUI lag
        # --------------------------------
        
        self.transform = T.Compose([
            T.ToPILImage(), 
            T.ToTensor()
        ])
        
        self.setup_ui()
        self.load_selected_model("Simple CNN (Custom)")

    def load_selected_model(self, model_name):
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
            elif model_name == "VDSR (Custom)":
                self.model = VDSR()
                weight_file = "vdsr_weights.pth"
            else:
                self.rest_output_view.setText(f"{model_name} is not yet implemented.\nPlease select a valid model.")
                self.rest_output_view.setStyleSheet("background: #ecf0f1; border: 3px dashed #e74c3c; color: #e74c3c;")
                self.model = None
                return

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
        
        # --- Color Mode & Blur Toggle ---
        mode_layout = QHBoxLayout()
        self.radio_rgb = QRadioButton("RGB")
        self.radio_gray = QRadioButton("Grayscale")
        self.radio_rgb.setChecked(True)
        self.check_blur = QCheckBox("Apply Artificial Blur")
        
        # Connect toggles to instantly update static images
        self.radio_rgb.toggled.connect(self.update_static_preview)
        self.radio_gray.toggled.connect(self.update_static_preview)
        self.check_blur.toggled.connect(self.update_static_preview)
        
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.radio_rgb)
        mode_layout.addWidget(self.radio_gray)
        mode_layout.addWidget(self.check_blur)
        mode_layout.addStretch()
        
        self.rest_input_view = QLabel("Live Input")
        self.rest_input_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rest_input_view.setStyleSheet("background: black; border: 2px solid #34495e;")
        self.rest_input_view.setMinimumSize(450, 350)
        
        # Update Input VOL Label for activity
        self.input_vol_lbl = QLabel("Input VOL: -- (Initializing)")
        self.input_vol_lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: blue;") # Blue means live feed
        self.input_vol_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left_layout.addLayout(btn_layout)
        left_layout.addLayout(mode_layout) 
        left_layout.addWidget(self.rest_input_view, stretch=1)
        left_layout.addWidget(self.input_vol_lbl)
        input_group.setLayout(left_layout)
        
        # --- RIGHT PANEL: AI OUTPUT ---
        output_group = QGroupBox("2. AI Restoration Module")
        output_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #7f8c8d; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; }")
        right_layout = QVBoxLayout()
        
        engine_layout = QHBoxLayout()
        engine_layout.addWidget(QLabel("Select AI Engine:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "Simple CNN (Custom)", 
            "SRCNN (Custom)", 
            "VDSR (Custom)", 
            "SwinIR (Upcoming)", 
            "Real-ESRGAN (Pre-trained)"
        ])
        self.model_selector.currentTextChanged.connect(self.load_selected_model)
        engine_layout.addWidget(self.model_selector)
        
        engine_layout.addWidget(QLabel("Resolution:"))
        self.scale_selector = QComboBox()
        self.scale_selector.addItems([
            "1x (Native 256x256)", 
            "2x (512x512 Tiled)", 
            "3x (768x768 Tiled)", 
            "4x (1024x1024 Tiled)"
        ])
        engine_layout.addWidget(self.scale_selector)

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

    # --- HELPER: Get Display Frame (Handles Blur & Grayscale) ---
    def get_display_frame(self):
        """Returns the frame with blur or grayscale applied based on UI."""
        if self.current_frame is None: return None
        
        # Make a copy so we don't permanently destroy the original live frame
        frame = self.current_frame.copy()
        
        # 1. Apply Artificial Blur if checked
        if self.check_blur.isChecked():
            frame = cv2.GaussianBlur(frame, (15, 15), 0)
            
        # 2. Apply Grayscale if checked
        if self.radio_gray.isChecked() and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        return frame

    def update_static_preview(self):
        """Instantly updates the preview if toggles are clicked while paused."""
        if self.current_frame is not None and not self.live_mode:
            display_frame = self.get_display_frame()
            pix_rest = self._convert_cv_to_qpixmap(display_frame, self.rest_input_view.width(), self.rest_input_view.height())
            self.rest_input_view.setPixmap(pix_rest)
            vol_in = calculate_vol(display_frame)
            self.input_vol_lbl.setText(f"Input VOL: {vol_in:.2f}")

    def update_live_feed(self, frame):
        if self.live_mode:
            self.current_frame = frame
            display_frame = self.get_display_frame() # Uses the toggle helper
            pix_rest = self._convert_cv_to_qpixmap(display_frame, self.rest_input_view.width(), self.rest_input_view.height())
            self.rest_input_view.setPixmap(pix_rest)

            # --- NEW: Live VOL Calculation for AI Tab ---
            current_time = time.time()
            if current_time - self.last_vol_time > self.vol_update_interval:
                live_vol = calculate_vol(display_frame)
                self.input_vol_lbl.setText(f"Live Input VOL: {live_vol:.2f}")
                
                # Matching main.py thresholds for visual status
                if live_vol > 1000:
                    self.input_vol_lbl.setStyleSheet("font-weight: bold; color: #27ae60;") # Green
                else:
                    self.input_vol_lbl.setStyleSheet("font-weight: bold; color: blue;") # Blue means live but blurred
                
                self.last_vol_time = current_time
            # ---------------------------------------------

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "Kayu", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            img = cv2.imread(file_name)
            if img is not None:
                self.live_mode = False 
                self.current_frame = img
                self.update_static_preview() # Instantly applies RGB/Gray and Blur
                
                self.rest_output_view.setText("Ready for Processing...")
                self.rest_output_view.setStyleSheet("background: #ecf0f1; border: 3px solid #f1c40f;")
                self.output_vol_lbl.setText("Output VOL: --")

    def resume_live(self):
        self.live_mode = True
        self.load_selected_model(self.model_selector.currentText()) 
        
        # Reset VOL label state for live feed activity
        self.input_vol_lbl.setText("Input VOL: -- (Live Feed)")
        self.input_vol_lbl.setStyleSheet("font-weight: bold; color: blue;") # indicate active state
        self.output_vol_lbl.setText("Output VOL: --")

    def _process_in_patches(self, img_bgr, tile_size=256):
        """Chops the image into 256x256 tiles, processes them, and stitches them back."""
        h, w, c = img_bgr.shape
        output_img = np.zeros_like(img_bgr)
        
        # Calculate how many tiles we need
        y_tiles = math.ceil(h / tile_size)
        x_tiles = math.ceil(w / tile_size)
        
        for y in range(y_tiles):
            for x in range(x_tiles):
                # Define tile boundaries
                y0, y1 = y * tile_size, min((y + 1) * tile_size, h)
                x0, x1 = x * tile_size, min((x + 1) * tile_size, w)
                
                # Extract tile
                tile = img_bgr[y0:y1, x0:x1]
                
                # Pad tile if it's smaller than 256x256 (happens at the edges)
                pad_h = tile_size - (y1 - y0)
                pad_w = tile_size - (x1 - x0)
                if pad_h > 0 or pad_w > 0:
                    tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                
                # Process the tile
                img_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                input_tensor = self.transform(img_rgb).unsqueeze(0)
                
                with torch.no_grad():
                    out_tensor = self.model(input_tensor)
                
                out_tile = out_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                out_tile = (np.clip(out_tile, 0, 1) * 255).astype(np.uint8)
                out_tile_bgr = cv2.cvtColor(out_tile, cv2.COLOR_RGB2BGR)
                
                # Crop the padding back off and place it in the final image
                output_img[y0:y1, x0:x1] = out_tile_bgr[0:(y1-y0), 0:(x1-x0)]
                
        return output_img

    def run_inference(self):
        if self.current_frame is None or not self.ai_available or getattr(self, 'model', None) is None:
            return
        
        try:
            # 1. Grab current display frame (handles Grayscale & Blur logic)
            display_frame = self.get_display_frame()
            
            # --- CRITICAL FIX FOR AI MODELS ---
            # Even if the image is Grayscale, the AI Model expects 3 channels.
            # We convert the 1-channel Gray back to a 3-channel BGR format.
            if len(display_frame.shape) == 2:
                ai_ready_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
            else:
                ai_ready_frame = display_frame

            # 2. Determine the target size based on the dropdown
            scale_text = self.scale_selector.currentText()
            if "1x" in scale_text: target_size = (256, 256)
            elif "2x" in scale_text: target_size = (512, 512)
            elif "3x" in scale_text: target_size = (768, 768)
            elif "4x" in scale_text: target_size = (1024, 1024)
            else: target_size = (256, 256)

            self.rest_output_view.setText("Processing Tiled Restoration...\n(This may take a moment)")
            self.rest_output_view.repaint()

            # 3. Resize input to target (This stretches it)
            upscaled_input = cv2.resize(ai_ready_frame, target_size, interpolation=cv2.INTER_CUBIC)
            
            # Update Input VOL to show what the baseline blur is at this size
            vol_in = calculate_vol(upscaled_input)
            
            # Change color to red to show this is the mathematical baseline, NOT the raw camera feed
            self.input_vol_lbl.setText(f"Baseline VOL: {vol_in:.2f} ({scale_text[:2]})")
            self.input_vol_lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: #e74c3c;") # Red baseline snapshot

            # 4. Process the large image using the 256x256 Tiling Method!
            output_bgr = self._process_in_patches(upscaled_input, tile_size=256)
            
            # 5. Final Metrics and Display
            vol_out = calculate_vol(output_bgr)
            self.output_vol_lbl.setText(f"Output VOL: {vol_out:.2f} ({scale_text[:2]})")
            
            pix = self._convert_cv_to_qpixmap(output_bgr, self.rest_output_view.width(), self.rest_output_view.height())
            self.rest_output_view.setPixmap(pix)
            self.rest_output_view.setStyleSheet("border: 3px solid #27ae60;") 
            
        except Exception as e:
            if "CUDA out of memory" in str(e):
                QMessageBox.warning(self, "GPU Memory Error", "Image too large for the GTX 1660 Ti!\nTry a smaller multiplier.")
            else:
                QMessageBox.critical(self, "AI Error", f"Inference failed: {str(e)}")

    def _convert_cv_to_qpixmap(self, frame, label_width, label_height):
        if len(frame.shape) == 2:
            h, w = frame.shape
            q_img = QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, ch = frame.shape
            q_img = QImage(frame.data, w, h, w * ch, QImage.Format.Format_BGR888)
        return QPixmap.fromImage(q_img).scaled(QSize(label_width, label_height), Qt.AspectRatioMode.KeepAspectRatio)