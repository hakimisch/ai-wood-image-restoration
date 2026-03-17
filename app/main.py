# app/main.py

import sys
import cv2
import os
import sqlite3
import numpy as np
import subprocess
import time
from datetime import datetime

# PyQt6 UI Imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QLineEdit, QLabel, QWidget, QComboBox, QHBoxLayout, 
                             QRadioButton, QFrame, QProgressBar, QMessageBox, QTabWidget)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap

# Custom Modules
from camera_thread import CameraThread
from ai import AIRestorationTab  # NEW: Import the extracted AI Tab
from training_tab import TrainingTab # NEW: Import the Training Tab

def calculate_vol(image):
    """Calculates the Variance of Laplacian (Sharpness Score)[cite: 20]."""
    if len(image.shape) == 2:
        return cv2.Laplacian(image, cv2.CV_64F).var()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

class MicroscopeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAIRO Image Restoration Suite - Acquisition & AI")
        self.setMinimumSize(1100, 850)
        self.setStyleSheet("background-color: #f0f0f0;") 

        # 1. Initialize State and Database
        self.init_db() 
        self.sub_image_count = 1 
        self.target_count = 20
        self.last_vol_time = 0
        self.vol_update_interval = 0.05 

        # 2. Setup Layout with Tabs [cite: 210]
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Tab 1: Data Acquisition ---
        self.acq_tab = QWidget()
        self.setup_acquisition_tab()
        self.tabs.addTab(self.acq_tab, "📸 Data Acquisition")

        # --- Tab 2: Training & Analytics (NEW) ---
        self.train_tab = TrainingTab()
        self.tabs.addTab(self.train_tab, "📈 Training & Analytics")

        # --- Tab 3: AI Restoration (Now an imported Module) ---
        self.rest_tab = AIRestorationTab()
        self.tabs.addTab(self.rest_tab, "⚡ AI Restoration")

        # 3. Global Threading and Camera Setup [cite: 43]
        self.thread = CameraThread() 
        self.thread.change_pixmap_signal.connect(self.update_all_feeds)
        self.camera_selector.currentIndexChanged.connect(self.change_camera)
        self.scan_cameras()

    def setup_acquisition_tab(self):
        """Layout for the standard dataset collection workflow (Stable Version)."""
        main_layout = QVBoxLayout()
        
        # --- Top Control Bar ---
        top_bar = QHBoxLayout()
        self.block_name_input = QLineEdit("") 
        self.block_name_input.setPlaceholderText("Enter Block ID (e.g. TB09)")
        self.block_name_input.setFixedWidth(150)
        self.block_name_input.textChanged.connect(self.validate_inputs)

        self.btn_reset = QPushButton("New Block")
        self.btn_reset.clicked.connect(self.reset_counter)

        self.radio_rgb = QRadioButton("RGB")
        self.radio_gray = QRadioButton("Gray")
        self.radio_rgb.setChecked(True)

        self.camera_selector = QComboBox()
        self.btn_scan = QPushButton("Scan")
        self.btn_scan.clicked.connect(self.scan_cameras)
        
        self.btn_open_folder = QPushButton("📂")
        self.btn_open_folder.clicked.connect(self.open_data_folder)

        self.live_vol_label = QLabel("Live VOL: 0.00")
        self.live_vol_label.setStyleSheet("font-weight: bold; color: #2980b9; padding: 5px;")

        self.btn_verify = QPushButton("Verify Dataset")
        self.btn_verify.clicked.connect(self.verify_dataset_integrity)
        self.verify_status_lbl = QLabel("Status: Ready")
        self.verify_status_lbl.setStyleSheet("color: gray; font-size: 11px;")

        top_bar.addWidget(QLabel("Block ID:"))
        top_bar.addWidget(self.block_name_input)
        top_bar.addWidget(self.btn_reset)
        top_bar.addStretch()
        top_bar.addWidget(self.radio_rgb)
        top_bar.addWidget(self.radio_gray)
        top_bar.addWidget(self.camera_selector)
        top_bar.addWidget(self.live_vol_label)
        top_bar.addWidget(self.btn_scan)
        top_bar.addWidget(self.btn_open_folder)
        top_bar.addWidget(self.btn_verify)

        # --- Species Registration Row [cite: 189] ---
        species_layout = QHBoxLayout()
        self.new_species_name = QLineEdit()
        self.new_species_name.setPlaceholderText("Species Name (e.g. Meranti)")
        self.new_species_initials = QLineEdit()
        self.new_species_initials.setPlaceholderText("Initials (e.g. ME)")
        self.new_species_scientific = QLineEdit()
        self.new_species_scientific.setPlaceholderText("Scientific Name (e.g. Koompassia excelsa)")
        
        self.btn_add_species = QPushButton("Save New Species")
        self.btn_add_species.clicked.connect(self.add_species_to_db)
        
        species_layout.addWidget(QLabel("New Species:"))
        species_layout.addWidget(self.new_species_name)
        species_layout.addWidget(self.new_species_initials)
        species_layout.addWidget(self.btn_add_species)
        species_layout.addWidget(QLabel("Scientific:"))
        species_layout.addWidget(self.new_species_scientific)

        # --- Main Live Feed View ---
        self.live_feed = QLabel("Initializing Feed...")
        self.live_feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_feed.setStyleSheet("background-color: #d3d3d3; border: 2px solid #555;")
        self.live_feed.setMinimumHeight(450)

        # --- Dataset Progress [cite: 47] ---
        prog_layout = QHBoxLayout()
        self.progress_label = QLabel(f"Block Progress: 0/{self.target_count}")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.target_count)
        self.progress_bar.setValue(0)
        prog_layout.addWidget(self.progress_label)
        prog_layout.addWidget(self.progress_bar)

        # --- Static Preview Area ---
        self.preview_frame = QFrame()
        self.preview_frame.setStyleSheet("background-color: white; border: 1px solid gray;")
        preview_frame_layout = QHBoxLayout()

        self.clear_prev = QLabel("Clear Preview")
        self.clear_prev.setMinimumSize(400, 300)
        self.vol_clear_lbl = QLabel("VOL Clear: --")

        self.blur_prev = QLabel("Blur Preview")
        self.blur_prev.setMinimumSize(400, 300)
        self.vol_blur_lbl = QLabel("VOL Blur: --")

        for container, label, vol_lbl, title in [(QVBoxLayout(), self.clear_prev, self.vol_clear_lbl, "Clear"), 
                                                 (QVBoxLayout(), self.blur_prev, self.vol_blur_lbl, "Blur")]:
            container.addWidget(QLabel(f"Last {title}:"))
            container.addWidget(label)
            container.addWidget(vol_lbl)
            preview_frame_layout.addLayout(container, stretch=1)
        self.preview_frame.setLayout(preview_frame_layout)

        # --- Capture Button ---
        self.btn_capture = QPushButton("CAPTURE & EVALUATE")
        self.btn_capture.setStyleSheet("background-color: #27ae60; color: white; padding: 15px; font-weight: bold;")
        self.btn_capture.setEnabled(False)
        self.btn_capture.clicked.connect(self.save_sample)

        # --- Assemble Layout ---
        main_layout.addLayout(top_bar)
        main_layout.addLayout(species_layout)
        main_layout.addWidget(self.live_feed, stretch=2)
        main_layout.addLayout(prog_layout)
        main_layout.addWidget(self.verify_status_lbl)
        main_layout.addWidget(self.preview_frame, stretch=3)
        main_layout.addWidget(self.btn_capture)
        
        self.acq_tab.setLayout(main_layout)

    def update_all_feeds(self, frame):
        """Pushes camera frames to both tabs and calculates VOL."""
        current_time = time.time()

        # Update Live Sharpness Metrics (Acquisition Tab)
        if current_time - self.last_vol_time > self.vol_update_interval:
            live_vol = calculate_vol(frame)
            self.live_vol_label.setText(f"Live VOL: {live_vol:.2f}")
            
            if live_vol > 1000:
                self.live_vol_label.setStyleSheet("font-weight: bold; color: #27ae60;")
            else:
                self.live_vol_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
            
            self.last_vol_time = current_time

        # Update Acquisition Tab Live Feed
        pix_acq = self._convert_cv_to_qpixmap(frame, self.live_feed.width(), 
                                               self.live_feed.height(), 
                                               self.radio_gray.isChecked())
        self.live_feed.setPixmap(pix_acq)

        # Update Restoration Tab Live Feed
        self.rest_tab.update_live_feed(frame)

    def save_sample(self):
        """Saves pairs and logs to DB[cite: 38]."""
        try:
            # Note: We now pull the frame directly from the UI since self.current_frame 
            # is no longer saved as an instance variable in main.py to save memory.
            # We can grab the frame directly from the camera thread's current state if needed,
            # but for simplicity, we rely on the threaded live feed.
            if self.thread.frame is None: return
            frame_to_save = self.thread.frame.copy()
            
            self.refresh_species_glossary() 
            
            # 1. Species Parsing
            full_input = self.block_name_input.text().strip().upper()
            initials = ""
            for i in range(len(full_input), 0, -1):
                if full_input[:i] in self.species_glossary:
                    initials = full_input[:i]
                    break
            
            species_name = self.species_glossary.get(initials, "Unknown")
            
            if species_name == "Unknown":
                QMessageBox.critical(self, "Species Error", 
                                     f"Could not identify species for '{full_input}'.\n"
                                     "Please register the species initials first.")
                return

            # 2. Dynamic Pathing
            block_dir = os.path.join("Kayu", species_name, full_input)
            clear_dir = os.path.join(block_dir, "clear")
            blur_dir = os.path.join(block_dir, "blur")
            os.makedirs(clear_dir, exist_ok=True)
            os.makedirs(blur_dir, exist_ok=True)

            # 3. Processing
            image_id = f"{full_input}{self.sub_image_count:04d}"
            mode = "GRAY" if self.radio_gray.isChecked() else "RGB"
            clear_img = frame_to_save
            
            if mode == "GRAY" and len(clear_img.shape) == 3:
                clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2GRAY)
            
            blur_img = cv2.GaussianBlur(clear_img, (15, 15), 0)
            v_clear = calculate_vol(clear_img)
            v_blur = calculate_vol(blur_img)

            # 4. Physical Saving
            cp = os.path.join(clear_dir, f"{image_id}.jpg")
            bp = os.path.join(blur_dir, f"{image_id}.jpg")
            cv2.imwrite(cp, clear_img)
            cv2.imwrite(bp, blur_img)

            # 5. UI Preview Update
            is_gray = (mode == "GRAY")
            self.clear_prev.setPixmap(self._convert_cv_to_qpixmap(clear_img, self.clear_prev.width(), self.clear_prev.height(), is_gray))
            self.blur_prev.setPixmap(self._convert_cv_to_qpixmap(blur_img, self.blur_prev.width(), self.blur_prev.height(), is_gray))
            self.vol_clear_lbl.setText(f"VOL Clear: {v_clear:.2f}")
            self.vol_blur_lbl.setText(f"VOL Blur: {v_blur:.2f}")
            self.progress_bar.setValue(min(self.sub_image_count, self.target_count))
            self.progress_label.setText(f"Block Progress: {self.sub_image_count}/{self.target_count}")

            # 6. Database Logging [cite: 102]
            with sqlite3.connect('data/database.db') as conn:
                conn.execute('''INSERT OR REPLACE INTO samples 
                    (species_name, species_initials, sample_name, mode, 
                     clear_path, blur_path, vol_clear, vol_blur, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                 (species_name, initials, image_id, mode, cp, bp, 
                  v_clear, v_blur, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()

            print(f"✅ Saved and Logged: {image_id} to {block_dir}")
            self.sub_image_count += 1

        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Image saved to disk, but DB failed: {str(e)}")

    # --- Helper Utilities ---

    def validate_inputs(self):
        self.btn_capture.setEnabled(len(self.block_name_input.text().strip()) > 0)

    def init_db(self):
        os.makedirs("data", exist_ok=True)
        with sqlite3.connect('data/database.db') as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS samples 
                        (id INTEGER PRIMARY KEY, species_name TEXT, species_initials TEXT,
                         sample_name TEXT UNIQUE, mode TEXT, clear_path TEXT, blur_path TEXT,
                         vol_clear REAL, vol_blur REAL, timestamp TEXT)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS species_registry 
                        (initials TEXT PRIMARY KEY, full_name TEXT, scientific_name TEXT)''')
        self.refresh_species_glossary()

    def refresh_species_glossary(self):
        conn = sqlite3.connect('data/database.db')
        cursor = conn.execute("SELECT initials, full_name FROM species_registry")
        self.species_glossary = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

    def add_species_to_db(self):
        name = self.new_species_name.text().strip()
        initials = self.new_species_initials.text().strip().upper()
        scientific = self.new_species_scientific.text().strip()

        if name and initials:
            try:
                with sqlite3.connect('data/database.db') as conn:
                    conn.execute("INSERT OR REPLACE INTO species_registry VALUES (?, ?, ?)", 
                                 (initials, name, scientific))
                self.refresh_species_glossary()
                QMessageBox.information(self, "Success", f"Registered: {name} ({scientific})")
                self.new_species_name.clear()
                self.new_species_initials.clear()
                self.new_species_scientific.clear()
            except Exception as e:
                QMessageBox.critical(self, "DB Error", f"Failed to add species: {e}")

    def reset_counter(self):
        self.sub_image_count = 1
        self.block_name_input.clear() 
        self.block_name_input.setFocus()
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Block Progress: 0/{self.target_count}")
        self.vol_clear_lbl.setText("VOL Clear: --")
        self.vol_blur_lbl.setText("VOL Blur: --")
        self.clear_prev.setText("Clear Preview")
        self.blur_prev.setText("Blur Preview")
        print("♻️ Reset: Awaiting new Block ID...")

    def open_data_folder(self):
        path = os.path.join(os.getcwd(), "Kayu")
        os.makedirs(path, exist_ok=True)
        subprocess.Popen(f'explorer "{path}"')

    def verify_dataset_integrity(self):
        self.verify_status_lbl.setText("Verifying... please wait.")
        self.verify_status_lbl.setStyleSheet("color: blue;")
        QApplication.processEvents() 

        try:
            conn = sqlite3.connect('data/database.db')
            cursor = conn.execute("SELECT sample_name, clear_path, blur_path FROM samples")
            rows = cursor.fetchall()
            
            missing_count = 0
            total_count = len(rows)

            for s_name, cp, bp in rows:
                full_cp = os.path.normpath(cp)
                full_bp = os.path.normpath(bp)

                if not os.path.exists(full_cp) or not os.path.exists(full_bp):
                    missing_count += 1
            
            conn.close()

            if missing_count == 0:
                self.verify_status_lbl.setText(f"✅ All {total_count} samples verified on disk.")
                self.verify_status_lbl.setStyleSheet("color: #27ae60;")
            else:
                self.verify_status_lbl.setText(f"❌ {missing_count}/{total_count} paths broken (Check Logs)")
                self.verify_status_lbl.setStyleSheet("color: #e74c3c;")

        except Exception as e:
            self.verify_status_lbl.setText(f"Error: {str(e)}")

    def scan_cameras(self):
        self.camera_selector.clear()
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_selector.addItem(f"Source {i}", i)
                cap.release()

    def change_camera(self, index):
        if index != -1:
            self.thread.update_index(self.camera_selector.itemData(index))

    def _convert_cv_to_qpixmap(self, frame, label_width, label_height, is_gray=False):
        if is_gray and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(frame.shape) == 2:
            h, w = frame.shape
            q_img = QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, ch = frame.shape
            q_img = QImage(frame.data, w, h, w * ch, QImage.Format.Format_BGR888)
        return QPixmap.fromImage(q_img).scaled(QSize(label_width, label_height), Qt.AspectRatioMode.KeepAspectRatio)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MicroscopeApp()
    win.show()
    sys.exit(app.exec())