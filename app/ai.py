# app/ai_tab.py
#
# AI Restoration Tab — also supports "Restore + Classify" pipeline:
# after running restoration inference, the restored image is passed
# through the species classifier (ResNet18) to measure downstream impact.

import cv2
import numpy as np
import os
import math
import time
import json
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QMessageBox, QFileDialog, QGroupBox,
                             QComboBox, QRadioButton, QCheckBox)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap

# AI/Machine Learning Imports
try:
    from models import SimpleRestorationNet, SRCNN, VDSR, SwinIR, RRDBNet
    import torch
    AI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    AI_AVAILABLE = False
    print("⚠️ AI modules not found. Restoration tab will be limited.")

# Classifier import (optional — for Restore + Classify pipeline)
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.classifier import create_classifier
    CLASSIFIER_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    CLASSIFIER_AVAILABLE = False
    print(f"⚠️ Classifier module not found: {e}")

import sys

def calculate_vol(image):
    if len(image.shape) == 2:
        return cv2.Laplacian(image, cv2.CV_64F).var()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# Device used for inference — set once at module level
_DEVICE = torch.device("cuda" if (AI_AVAILABLE and torch.cuda.is_available()) else "cpu") if AI_AVAILABLE else None

class AIRestorationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.ai_available  = AI_AVAILABLE
        self.live_mode     = True
        self.device        = _DEVICE

        self.last_vol_time      = 0
        self.vol_update_interval = 0.05

        # Classifier for Restore + Classify pipeline
        self.classifier = None
        self.classifier_loaded = False
        self._load_classifier()

        # No PIL transform needed — _to_tensor() does it directly
        self.setup_ui()
        self.load_selected_model()

    # ── Classifier Loading (for Restore + Classify) ────────────────────────
    DEFAULT_CLASSIFIER_WEIGHTS = "classifier_weights.pth"

    def _load_classifier(self, weights_path=None):
        """Load the species classifier for post-restoration classification."""
        if not CLASSIFIER_AVAILABLE:
            return

        if weights_path is None:
            weights_path = self.DEFAULT_CLASSIFIER_WEIGHTS

        if not os.path.exists(weights_path):
            print(f"⚠️ Classifier weights not found at '{weights_path}' — Restore+Classify disabled.")
            self.classifier_loaded = False
            if hasattr(self, 'btn_restore_classify'):
                self.btn_restore_classify.setEnabled(False)
                self.btn_restore_classify.setToolTip(
                    "Train the classifier first via Training > Classifier Training tab"
                )
            return

        try:
            self.classifier = create_classifier(weights_path=weights_path)
            self.classifier_loaded = True
            print(f"🔬 Classifier loaded for Restore+Classify ({self.classifier.num_species} species)")
            if hasattr(self, 'btn_restore_classify'):
                self.btn_restore_classify.setEnabled(True)
                self.btn_restore_classify.setToolTip(
                    "Run restoration then classify the restored output"
                )
        except Exception as e:
            print(f"⚠️ Failed to load classifier: {e}")
            self.classifier_loaded = False
            if hasattr(self, 'btn_restore_classify'):
                self.btn_restore_classify.setEnabled(False)

    @staticmethod
    def _to_tensor(img_bgr):
        """BGR uint8 numpy → CHW float32 [0,1] tensor, no PIL."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0

    def refresh_pth_list(self):
        """Scans the root directory for .pth files and updates the dropdown."""
        current_selection = self.pth_selector.currentText()
        self.pth_selector.blockSignals(True)
        self.pth_selector.clear()
        
        pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        
        if not pth_files:
            self.pth_selector.addItem("No .pth files found")
        else:
            self.pth_selector.addItems(pth_files)
            
        if current_selection in pth_files:
            self.pth_selector.setCurrentText(current_selection)
            
        self.pth_selector.blockSignals(False)

    def load_selected_model(self, *args):
        """Loads the architecture from the Model selector and weights from the PTH selector."""
        if not self.ai_available: return
        
        model_name = self.model_selector.currentText()
        weight_file = self.pth_selector.currentText()
        
        try:
            self.rest_output_view.setText(f"Loading {model_name}...")
            self.rest_output_view.setStyleSheet("background: #ecf0f1; border: 3px dashed #bdc3c7; color: #f39c12;")
            
            # 1. Initialize Architecture
            if model_name == "Simple CNN (Custom)":
                self.model = SimpleRestorationNet()
            elif model_name == "SRCNN (Custom)":
                self.model = SRCNN()
            elif model_name == "VDSR (Custom)":
                self.model = VDSR()
            elif model_name == "SwinIR (Custom)":
                self.model = SwinIR(img_size=128)
            elif "Real-ESRGAN" in model_name:
                self.model = RRDBNet()
            else:
                self.rest_output_view.setText(f"{model_name} is not yet implemented.\nPlease select a valid model.")
                self.rest_output_view.setStyleSheet("background: #ecf0f1; border: 3px dashed #e74c3c; color: #e74c3c;")
                self.model = None
                return

            # 2. Inject Weights
            if weight_file and weight_file.endswith('.pth') and os.path.exists(weight_file):
                try:
                    self.model.load_state_dict(
                        torch.load(weight_file, map_location='cpu', weights_only=True)
                    )
                    self.rest_output_view.setText(f"✅ {model_name} Loaded\nWeights: {weight_file}")
                    self.rest_output_view.setStyleSheet("background: #ecf0f1; border: 3px dashed #27ae60; color: #27ae60;")
                except Exception as e:
                    self.rest_output_view.setText(f"⚠️ Architecture Mismatch!\n'{weight_file}' does not match {model_name}.")
                    self.rest_output_view.setStyleSheet("background: #ecf0f1; border: 3px dashed #e74c3c; color: #e74c3c;")
                    print(f"Error loading weights: {e}")
            else:
                self.rest_output_view.setText(f"⚠️ No weights loaded.\nRunning {model_name} with blank weights.")
                self.rest_output_view.setStyleSheet("background: #ecf0f1; border: 3px dashed #e67e22; color: #e67e22;")

            if getattr(self, 'model', None) is not None:
                # Move to GPU for inference — previously ran on CPU only
                self.model = self.model.to(self.device)
                self.model.eval()
            print(f"✨ Switched AI Engine to: {model_name} | Weights: {weight_file} | Device: {self.device}")
            
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
        
        mode_layout = QHBoxLayout()
        self.radio_rgb = QRadioButton("RGB")
        self.radio_gray = QRadioButton("Grayscale")
        self.radio_rgb.setChecked(True)
        self.check_blur = QCheckBox("Apply Artificial Blur")
        
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
        
        self.input_vol_lbl = QLabel("Input VOL: -- (Initializing)")
        self.input_vol_lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: blue;")
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
        engine_layout.addWidget(QLabel("Architecture:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "Simple CNN (Custom)", 
            "SRCNN (Custom)", 
            "VDSR (Custom)", 
            "SwinIR (Custom)", 
            "Real-ESRGAN (Custom)"
        ])
        self.model_selector.currentTextChanged.connect(self.load_selected_model)
        engine_layout.addWidget(self.model_selector)
        
        engine_layout.addWidget(QLabel("Scale:"))
        self.scale_selector = QComboBox()
        self.scale_selector.addItems([
            "1x (Native 256x256)", 
            "2x (512x512 Tiled)", 
            "3x (768x768 Tiled)", 
            "4x (1024x1024 Tiled)"
        ])
        engine_layout.addWidget(self.scale_selector)

        # --- NEW: PTH File Selector ---
        weights_layout = QHBoxLayout()
        weights_layout.addWidget(QLabel("Weights File:"))
        self.pth_selector = QComboBox()
        self.refresh_pth_list()
        self.pth_selector.currentTextChanged.connect(self.load_selected_model)
        weights_layout.addWidget(self.pth_selector, stretch=1)
        
        self.btn_refresh_pth = QPushButton("🔄 Refresh")
        self.btn_refresh_pth.clicked.connect(self.refresh_pth_list)
        weights_layout.addWidget(self.btn_refresh_pth)

        self.btn_reload_classifier = QPushButton("🔬 Reload Classifier")
        self.btn_reload_classifier.clicked.connect(self._load_classifier)
        self.btn_reload_classifier.setToolTip("Reload classifier weights after training")
        weights_layout.addWidget(self.btn_reload_classifier)
        # -------------------------------

        self.rest_output_view = QLabel("AI Restored Result\n(Awaiting Inference)")
        self.rest_output_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rest_output_view.setStyleSheet("background: #ecf0f1; border: 3px dashed #bdc3c7; color: #7f8c8d;")
        self.rest_output_view.setMinimumSize(450, 350)
        
        self.output_vol_lbl = QLabel("Output VOL: --")
        self.output_vol_lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: #27ae60;")
        self.output_vol_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.vol_delta_lbl = QLabel("")
        self.vol_delta_lbl.setStyleSheet("font-size: 12px; color: #7f8c8d;")
        self.vol_delta_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.inference_time_lbl = QLabel("")
        self.inference_time_lbl.setStyleSheet("font-size: 11px; color: #95a5a6; font-style: italic;")
        self.inference_time_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_restore = QPushButton("✨ RUN AI RESTORATION")
        self.btn_restore.setMinimumHeight(50)
        self.btn_restore.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; font-size: 14px;")
        self.btn_restore.clicked.connect(self.run_inference)

        # ── Restore + Classify pipeline ────────────────────────────────────
        self.btn_restore_classify = QPushButton("🔬 RESTORE + CLASSIFY")
        self.btn_restore_classify.setMinimumHeight(50)
        self.btn_restore_classify.setStyleSheet(
            "background-color: #8e44ad; color: white; font-weight: bold; font-size: 14px;"
        )
        self.btn_restore_classify.clicked.connect(self.run_restore_and_classify)

        # Classification result display
        self.classify_result_label = QLabel("")
        self.classify_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.classify_result_label.setStyleSheet(
            "font-size: 14px; color: #7f8c8d; font-style: italic; padding: 4px;"
        )
        self.classify_top3_label = QLabel("")
        self.classify_top3_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.classify_top3_label.setStyleSheet(
            "font-size: 11px; color: #95a5a6;"
        )

        if not self.classifier_loaded:
            self.btn_restore_classify.setEnabled(False)
            self.btn_restore_classify.setToolTip(
                "Train the classifier first via Training > Classifier Training tab"
            )
        # ────────────────────────────────────────────────────────────────────

        if not self.ai_available:
            self.btn_restore.setEnabled(False)
            self.model_selector.setEnabled(False)
            self.pth_selector.setEnabled(False)

        right_layout.addLayout(engine_layout)
        right_layout.addLayout(weights_layout)
        right_layout.addWidget(self.rest_output_view, stretch=1)
        right_layout.addWidget(self.output_vol_lbl)
        right_layout.addWidget(self.vol_delta_lbl)
        right_layout.addWidget(self.inference_time_lbl)
        right_layout.addWidget(self.btn_restore)
        right_layout.addWidget(self.btn_restore_classify)
        right_layout.addWidget(self.classify_result_label)
        right_layout.addWidget(self.classify_top3_label)
        output_group.setLayout(right_layout)
        
        main_layout.addWidget(input_group, stretch=1)
        main_layout.addWidget(output_group, stretch=1)
        self.setLayout(main_layout)

    def get_display_frame(self):
        if self.current_frame is None: return None
        frame = self.current_frame.copy()
        if self.check_blur.isChecked():
            frame = cv2.GaussianBlur(frame, (15, 15), 0)
        if self.radio_gray.isChecked() and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def update_static_preview(self):
        if self.current_frame is not None and not self.live_mode:
            display_frame = self.get_display_frame()
            pix_rest = self._convert_cv_to_qpixmap(display_frame, self.rest_input_view.width(), self.rest_input_view.height())
            self.rest_input_view.setPixmap(pix_rest)
            vol_in = calculate_vol(display_frame)
            self.input_vol_lbl.setText(f"Input VOL: {vol_in:.2f}")

    def update_live_feed(self, frame):
        if self.live_mode:
            self.current_frame = frame
            display_frame = self.get_display_frame()
            pix_rest = self._convert_cv_to_qpixmap(display_frame, self.rest_input_view.width(), self.rest_input_view.height())
            self.rest_input_view.setPixmap(pix_rest)

            current_time = time.time()
            if current_time - self.last_vol_time > self.vol_update_interval:
                live_vol = calculate_vol(display_frame)
                self.input_vol_lbl.setText(f"Live Input VOL: {live_vol:.2f}")
                if live_vol > 1000:
                    self.input_vol_lbl.setStyleSheet("font-weight: bold; color: #27ae60;")
                else:
                    self.input_vol_lbl.setStyleSheet("font-weight: bold; color: blue;")
                self.last_vol_time = current_time

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "Kayu", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            img = cv2.imread(file_name)
            if img is not None:
                self.live_mode = False
                self.current_frame = img
                self.update_static_preview()
                self.rest_output_view.setText("Ready for Processing...")
                self.rest_output_view.setStyleSheet("background: #ecf0f1; border: 3px solid #f1c40f;")
                self.output_vol_lbl.setText("Output VOL: --")
                self.classify_result_label.setText("")
                self.classify_top3_label.setText("")

    def resume_live(self):
        self.live_mode = True
        self.load_selected_model()
        self.input_vol_lbl.setText("Input VOL: -- (Live Feed)")
        self.input_vol_lbl.setStyleSheet("font-weight: bold; color: blue;")
        self.output_vol_lbl.setText("Output VOL: --")
        self.classify_result_label.setText("")
        self.classify_top3_label.setText("")

    def _process_in_patches(self, img_bgr, tile_size=256):
        """Tile with overlap and blend to eliminate seam artifacts at tile boundaries.

        Each tile is processed with an `overlap` pixel border on all sides that
        it shares with a neighbour.  The results are blended using a linear
        weight ramp so that the centre of each tile has full weight and the
        edges taper to zero, making joins invisible.

        Overlap is set to 25% of tile_size (16px for 64px tiles, 64px for
        256px tiles) — enough to cover any border artifact the model creates.
        """
        h, w, c  = img_bgr.shape
        overlap  = tile_size // 4          # 16px for SwinIR, 64px for others
        step     = tile_size - overlap     # stride between tile origins

        # Accumulator arrays for blended output and weight sum
        output_acc  = np.zeros((h, w, c), dtype=np.float32)
        weight_acc  = np.zeros((h, w, 1),  dtype=np.float32)

        # Build a 2-D weight mask: linear ramp from 0→1→0 on each axis,
        # so the tile centre is weighted 1.0 and the overlapping edges taper
        ramp   = np.linspace(0, 1, overlap, endpoint=False, dtype=np.float32)
        ones   = np.ones(tile_size - 2 * overlap, dtype=np.float32)
        ramp_1d = np.concatenate([ramp, ones, ramp[::-1]])[:tile_size]
        weight_2d = np.outer(ramp_1d, ramp_1d)[:, :, np.newaxis]  # (T,T,1)

        y_starts = list(range(0, h - tile_size + 1, step))
        x_starts = list(range(0, w - tile_size + 1, step))
        # Always include a tile that reaches the bottom/right edge
        if not y_starts or y_starts[-1] + tile_size < h:
            y_starts.append(max(0, h - tile_size))
        if not x_starts or x_starts[-1] + tile_size < w:
            x_starts.append(max(0, w - tile_size))

        for y0 in y_starts:
            y1 = min(y0 + tile_size, h)
            for x0 in x_starts:
                x1 = min(x0 + tile_size, w)

                tile    = img_bgr[y0:y1, x0:x1]
                th, tw  = tile.shape[:2]

                # Pad to full tile_size if near an edge
                pad_h = tile_size - th
                pad_w = tile_size - tw
                if pad_h > 0 or pad_w > 0:
                    tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

                input_tensor = self._to_tensor(tile).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out_tensor = self.model(input_tensor)

                out_tile = out_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                out_tile = np.clip(out_tile, 0, 1)
                out_tile_bgr = cv2.cvtColor(
                    (out_tile * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                ).astype(np.float32)

                # Clone the standard weight mask so we can alter it for edge cases
                w2d = weight_2d.copy()

                # Prevent dark borders: If a tile touches the absolute image edge, 
                # force that side of the weight mask to 1.0
                if y0 == 0:                  w2d[:overlap, :] = 1.0
                if y1 == h:                  w2d[-overlap:, :] = 1.0
                if x0 == 0:                  w2d[:, :overlap] = 1.0
                if x1 == w:                  w2d[:, -overlap:] = 1.0

                w2d = w2d[:th, :tw] # Crop if padded

                # Blend only the valid (non-padded) region
                output_acc[y0:y1, x0:x1] += out_tile_bgr[:th, :tw] * w2d
                weight_acc[y0:y1, x0:x1] += w2d

        # Normalise by accumulated weights (avoid divide-by-zero in corners)
        weight_acc = np.maximum(weight_acc, 1e-6)
        result = (output_acc / weight_acc).clip(0, 255).astype(np.uint8)
        return result

    def run_inference(self):
        if self.current_frame is None or not self.ai_available or getattr(self, 'model', None) is None:
            return

        try:
            display_frame = self.get_display_frame()

            if len(display_frame.shape) == 2:
                ai_ready_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
            else:
                ai_ready_frame = display_frame

            scale_text = self.scale_selector.currentText()
            if   "1x" in scale_text: scale_factor = 1
            elif "2x" in scale_text: scale_factor = 2
            elif "3x" in scale_text: scale_factor = 3
            elif "4x" in scale_text: scale_factor = 4
            else:                    scale_factor = 1

            self.rest_output_view.setText("Processing AI Restoration...\n(This may take a moment)")
            self.rest_output_view.repaint()

            vol_in_raw = calculate_vol(ai_ready_frame)

            # Determine tile size dynamically
            if "SwinIR" in self.model_selector.currentText():
                t_size = 128
            elif "Real-ESRGAN" in self.model_selector.currentText():
                t_size = 96
            else:
                t_size = 256

            # STEP 1: Run AI Inference at Native 1x Scale First
            t_start = time.perf_counter()
            restored_1x_bgr = self._process_in_patches(ai_ready_frame, tile_size=t_size)
            t_elapsed = time.perf_counter() - t_start

            # ------------------------------------------------------------------
            # NEW: Luminance Transfer (Fixes ESRGAN Color Drift / Chroma Noise)
            # ------------------------------------------------------------------
            # Convert both the original input and the AI output to YCrCb
            orig_ycc = cv2.cvtColor(ai_ready_frame, cv2.COLOR_BGR2YCrCb)
            ai_ycc   = cv2.cvtColor(restored_1x_bgr, cv2.COLOR_BGR2YCrCb)

            # Swap the AI's hallucinated colors with the original pristine colors.
            # Y (Index 0) = Luminance (Sharpness) -> Kept from AI
            # Cr/Cb (Index 1 & 2) = Chrominance (Color) -> Replaced with Original
            ai_ycc[:, :, 1] = orig_ycc[:, :, 1] 
            ai_ycc[:, :, 2] = orig_ycc[:, :, 2]

            # Convert back to standard BGR for the UI
            restored_1x_bgr = cv2.cvtColor(ai_ycc, cv2.COLOR_YCrCb2BGR)
            # ------------------------------------------------------------------

            # STEP 2: Upscale the Pristine AI Output (if requested)
            if scale_factor > 1:
                orig_h, orig_w = restored_1x_bgr.shape[:2]
                final_output_bgr = cv2.resize(
                    restored_1x_bgr, 
                    (orig_w * scale_factor, orig_h * scale_factor),
                    interpolation=cv2.INTER_LANCZOS4
                )
            else:
                final_output_bgr = restored_1x_bgr

            vol_out = calculate_vol(final_output_bgr)
            delta   = vol_out - vol_in_raw
            sign    = "+" if delta >= 0 else ""

            self.input_vol_lbl.setText(f"Input VOL: {vol_in_raw:.1f}")
            self.input_vol_lbl.setStyleSheet("font-weight: bold; font-size: 13px; color: #e74c3c;")

            self.output_vol_lbl.setText(f"Output VOL: {vol_out:.1f} ({scale_factor}x)")
            self.vol_delta_lbl.setText(f"Overall gain vs raw input: {sign}{delta:.1f}")
            
            overlap   = t_size // 4
            step      = t_size - overlap
            n_y = len(list(range(0, ai_ready_frame.shape[0] - t_size + 1, step))) + 1
            n_x = len(list(range(0, ai_ready_frame.shape[1] - t_size + 1, step))) + 1
            
            self.inference_time_lbl.setText(
                f"Inference: {t_elapsed:.2f}s  |  "
                f"Tiles: {n_y * n_x} ({t_size}px)  |  "
                f"Device: {self.device}"
            )

            # Colour the output VOL label by how it compares to raw input
            if vol_out > vol_in_raw * 1.1:
                self.output_vol_lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: #27ae60;")
            elif vol_out > vol_in_raw * 0.9:
                self.output_vol_lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: #f39c12;")
            else:
                self.output_vol_lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: #e74c3c;")

            pix = self._convert_cv_to_qpixmap(final_output_bgr, self.rest_output_view.width(), self.rest_output_view.height())
            self.rest_output_view.setPixmap(pix)
            self.rest_output_view.setStyleSheet("border: 3px solid #27ae60;")

        except Exception as e:
            if "CUDA out of memory" in str(e):
                QMessageBox.warning(self, "GPU Memory Error", "Image too large for the GTX 1660 Ti!\nTry a smaller scale.")
            else:
                QMessageBox.critical(self, "AI Error", f"Inference failed: {str(e)}")

    # ── Restore + Classify Pipeline ────────────────────────────────────────

    def run_restore_and_classify(self):
        """Run restoration inference, then classify the restored output.

        This implements the thesis experiment:
            Blurry Input → SwinIR Restore → ResNet18 Classify
        and compares the result against classifying the un-restored input.
        """
        if self.current_frame is None or not self.ai_available:
            return
        if getattr(self, 'model', None) is None:
            QMessageBox.warning(self, "No Model", "Load a restoration model first.")
            return
        if not self.classifier_loaded:
            QMessageBox.warning(
                self, "No Classifier",
                "Classifier not loaded.\nTrain one in Training > Classifier Training tab."
            )
            return

        try:
            display_frame = self.get_display_frame()

            if len(display_frame.shape) == 2:
                ai_ready_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
            else:
                ai_ready_frame = display_frame

            # ── Step 1: Classify the UN-restored input (baseline) ────────
            raw_species, raw_confidence, raw_top3 = self.classifier.predict(
                ai_ready_frame, top_k=3
            )

            # ── Step 2: Run restoration ──────────────────────────────────
            scale_text = self.scale_selector.currentText()
            if   "1x" in scale_text: scale_factor = 1
            elif "2x" in scale_text: scale_factor = 2
            elif "3x" in scale_text: scale_factor = 3
            elif "4x" in scale_text: scale_factor = 4
            else:                    scale_factor = 1

            if "SwinIR" in self.model_selector.currentText():
                t_size = 128
            elif "Real-ESRGAN" in self.model_selector.currentText():
                t_size = 96
            else:
                t_size = 256

            self.rest_output_view.setText("Restoring + Classifying...\n(This may take a moment)")
            self.rest_output_view.repaint()

            t_start = time.perf_counter()
            restored_bgr = self._process_in_patches(ai_ready_frame, tile_size=t_size)
            t_elapsed = time.perf_counter() - t_start

            # Luminance transfer (same as run_inference)
            orig_ycc = cv2.cvtColor(ai_ready_frame, cv2.COLOR_BGR2YCrCb)
            ai_ycc   = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2YCrCb)
            ai_ycc[:, :, 1] = orig_ycc[:, :, 1]
            ai_ycc[:, :, 2] = orig_ycc[:, :, 2]
            restored_bgr = cv2.cvtColor(ai_ycc, cv2.COLOR_YCrCb2BGR)

            if scale_factor > 1:
                orig_h, orig_w = restored_bgr.shape[:2]
                final_output_bgr = cv2.resize(
                    restored_bgr,
                    (orig_w * scale_factor, orig_h * scale_factor),
                    interpolation=cv2.INTER_LANCZOS4
                )
            else:
                final_output_bgr = restored_bgr

            # ── Step 3: Classify the RESTORED output ─────────────────────
            restored_species, restored_confidence, restored_top3 = self.classifier.predict(
                final_output_bgr, top_k=3
            )

            # ── Step 4: Display results ──────────────────────────────────
            vol_in  = calculate_vol(ai_ready_frame)
            vol_out = calculate_vol(final_output_bgr)
            delta   = vol_out - vol_in
            sign    = "+" if delta >= 0 else ""

            self.input_vol_lbl.setText(f"Input VOL: {vol_in:.1f}")
            self.output_vol_lbl.setText(f"Output VOL: {vol_out:.1f} ({scale_factor}x)")
            self.vol_delta_lbl.setText(f"VOL gain: {sign}{delta:.1f}")
            self.inference_time_lbl.setText(
                f"Inference: {t_elapsed:.2f}s  |  Device: {self.device}"
            )

            # Classification comparison
            raw_top3_str = " | ".join([f"{s}: {c:.0%}" for s, c in raw_top3])
            rest_top3_str = " | ".join([f"{s}: {c:.0%}" for s, c in restored_top3])

            self.classify_result_label.setText(
                f"🔬 Before: {raw_species} ({raw_confidence:.1%})  →  "
                f"After Restore: {restored_species} ({restored_confidence:.1%})"
            )

            # Color-code: green if restored improved or matches, red if worse
            if restored_confidence >= raw_confidence:
                self.classify_result_label.setStyleSheet(
                    "font-size: 15px; font-weight: bold; color: #27ae60; padding: 4px;"
                )
            else:
                self.classify_result_label.setStyleSheet(
                    "font-size: 15px; font-weight: bold; color: #e67e22; padding: 4px;"
                )

            self.classify_top3_label.setText(
                f"Before Top-3: {raw_top3_str}  |  "
                f"After Top-3: {rest_top3_str}"
            )
            self.classify_top3_label.setStyleSheet(
                "font-size: 11px; color: #7f8c8d;"
            )

            # Show restored image in output view
            pix = self._convert_cv_to_qpixmap(
                final_output_bgr, self.rest_output_view.width(), self.rest_output_view.height()
            )
            self.rest_output_view.setPixmap(pix)
            self.rest_output_view.setStyleSheet("border: 3px solid #8e44ad;")

        except Exception as e:
            if "CUDA out of memory" in str(e):
                QMessageBox.warning(self, "GPU Memory Error",
                                    "Image too large for the GTX 1660 Ti!\nTry a smaller scale.")
            else:
                QMessageBox.critical(self, "Pipeline Error",
                                     f"Restore + Classify failed:\n{str(e)}")
                import traceback
                traceback.print_exc()

    def _convert_cv_to_qpixmap(self, frame, label_width, label_height):
        if len(frame.shape) == 2:
            h, w = frame.shape
            q_img = QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, ch = frame.shape
            q_img = QImage(frame.data, w, h, w * ch, QImage.Format.Format_BGR888)
        return QPixmap.fromImage(q_img).scaled(QSize(label_width, label_height), Qt.AspectRatioMode.KeepAspectRatio)