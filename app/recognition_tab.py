# app/recognition_tab.py
#
# Species Recognition Tab — Live camera classification + batch processing + analytics.
#
# Features:
#   - Live classification of the camera feed with species name + confidence overlay
#   - Batch classification of all unclassified images in the database
#   - Species analytics dashboard (per-species accuracy, confusion matrix)
#   - Export results to CSV
#
# Dependencies:
#   - classifier.py (SpeciesClassifier model)
#   - PyQt6, OpenCV, torch, matplotlib, numpy

import os
import sys
import json
import csv
import time
import sqlite3
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QFileDialog, QGroupBox, QComboBox, QProgressBar,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QCheckBox, QSpinBox, QTabWidget,
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont

# AI imports
try:
    from classifier import create_classifier
    import torch
    CLASSIFIER_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    CLASSIFIER_AVAILABLE = False
    print(f"⚠️ Classifier module not found: {e}")

# Matplotlib for analytics
try:
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ matplotlib not installed. Analytics charts will be disabled.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB_PATH = 'data/database.db'
DEFAULT_WEIGHTS = 'classifier_weights.pth'


# ---------------------------------------------------------------------------
# Batch Classification Thread
# ---------------------------------------------------------------------------

class BatchClassificationThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, classifier, db_path=DB_PATH, confidence_threshold=0.5):
        super().__init__()
        self.classifier = classifier
        self.db_path = db_path
        self.confidence_threshold = confidence_threshold

    def run(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all samples that don't have a classification yet
            cursor.execute("""
                SELECT s.sample_name, s.clear_path
                FROM samples s
                LEFT JOIN classifications c ON s.sample_name = c.sample_name
                WHERE c.sample_name IS NULL
            """)
            rows = cursor.fetchall()
            total = len(rows)
            conn.close()

            if total == 0:
                self.log_signal.emit("✅ All images already classified.")
                self.finished_signal.emit()
                return

            self.log_signal.emit(f"🔍 Batch classifying {total} images...")

            for i, (sample_name, clear_path) in enumerate(rows):
                if not os.path.exists(clear_path):
                    self.log_signal.emit(f"  ⚠️  Image not found: {clear_path}")
                    continue

                img_bgr = cv2.imread(clear_path)
                if img_bgr is None:
                    continue

                species, confidence, top3 = self.classifier.predict(img_bgr, top_k=3)

                # Only save if confidence meets threshold
                if confidence >= self.confidence_threshold:
                    conn = sqlite3.connect(self.db_path)
                    conn.execute(
                        "INSERT INTO classifications "
                        "(sample_name, predicted_species, confidence, top3_predictions, model_name, timestamp) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (sample_name, species, confidence,
                         json.dumps([{"species": s, "confidence": c} for s, c in top3]),
                         "ResNet18",
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    )
                    conn.commit()
                    conn.close()

                if (i + 1) % 50 == 0 or (i + 1) == total:
                    self.log_signal.emit(f"  [{i+1}/{total}] {species} ({confidence:.1%})")
                    self.progress_signal.emit(int((i + 1) / total * 100))

            self.log_signal.emit(f"\n✅ Batch classification complete! {total} images processed.")
        except Exception as e:
            self.log_signal.emit(f"❌ Batch classification error: {str(e)}")
        finally:
            self.finished_signal.emit()


# ---------------------------------------------------------------------------
# Analytics Tab (embedded within RecognitionTab)
# ---------------------------------------------------------------------------

class AnalyticsWidget(QWidget):
    """Embedded analytics dashboard showing species stats and confusion matrix."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Refresh button
        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("🔄 Refresh Analytics")
        self.btn_refresh.clicked.connect(self.refresh)
        btn_row.addWidget(self.btn_refresh)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Stats table
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(6)
        self.stats_table.setHorizontalHeaderLabels([
            "Species", "Total Images", "Avg VOL Clear", "Avg VOL Blur",
            "Classified", "Accuracy"
        ])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.stats_table, stretch=2)

        # Matplotlib chart area
        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(8, 4))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas, stretch=1)
        else:
            self.chart_placeholder = QLabel("Install matplotlib for analytics charts:\n  pip install matplotlib")
            self.chart_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.chart_placeholder.setStyleSheet("color: #7f8c8d; font-style: italic;")
            layout.addWidget(self.chart_placeholder)

        self.setLayout(layout)

    def refresh(self):
        """Reload data from DB and update table + chart."""
        if not os.path.exists(DB_PATH):
            return

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Species stats
        cursor.execute("""
            SELECT s.species_name,
                   COUNT(*) as total,
                   ROUND(AVG(s.vol_clear), 1),
                   ROUND(AVG(s.vol_blur), 1),
                   COUNT(DISTINCT c.id) as classified,
                   ROUND(AVG(CASE WHEN c.predicted_species = s.species_name THEN 1.0 ELSE 0 END), 3)
            FROM samples s
            LEFT JOIN classifications c ON s.sample_name = c.sample_name
            GROUP BY s.species_name
            ORDER BY s.species_name
        """)
        rows = cursor.fetchall()
        conn.close()

        self.stats_table.setRowCount(len(rows))
        species_names = []
        accuracies = []

        for i, (name, total, avg_clear, avg_blur, classified, acc) in enumerate(rows):
            self.stats_table.setItem(i, 0, QTableWidgetItem(name))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(total)))
            self.stats_table.setItem(i, 2, QTableWidgetItem(str(avg_clear)))
            self.stats_table.setItem(i, 3, QTableWidgetItem(str(avg_blur)))
            self.stats_table.setItem(i, 4, QTableWidgetItem(str(classified)))
            acc_item = QTableWidgetItem(f"{acc:.1%}" if acc else "N/A")
            if acc and acc > 0.8:
                acc_item.setForeground(Qt.GlobalColor.darkGreen)
            elif acc and acc > 0.5:
                acc_item.setForeground(Qt.GlobalColor.darkYellow)
            self.stats_table.setItem(i, 5, acc_item)
            species_names.append(name)
            accuracies.append(acc if acc else 0)

        # Update chart
        if MATPLOTLIB_AVAILABLE and species_names:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            colors = ['#27ae60' if a >= 0.8 else '#f39c12' if a >= 0.5 else '#e74c3c' for a in accuracies]
            bars = ax.bar(range(len(species_names)), accuracies, color=colors)
            ax.set_xticks(range(len(species_names)))
            ax.set_xticklabels(species_names, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Accuracy')
            ax.set_title('Per-Species Classification Accuracy')
            ax.set_ylim(0, 1.05)
            ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80% target')
            ax.legend(fontsize=8)
            self.figure.tight_layout()
            self.canvas.draw()


# ---------------------------------------------------------------------------
# Main Recognition Tab
# ---------------------------------------------------------------------------

class RecognitionTab(QWidget):
    """Main tab for species recognition — live, batch, and analytics."""

    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.live_mode = True
        self.classifier = None
        self.classifier_loaded = False

        # Timing for live classification throttling
        self.last_classify_time = 0
        self.classify_interval = 0.5  # seconds between live classifications

        self.setup_ui()
        self.load_classifier()

    # ── Classifier Loading ────────────────────────────────────────────────

    def load_classifier(self, weights_path=None):
        """Attempt to load the classifier model."""
        if not CLASSIFIER_AVAILABLE:
            self.recog_status.setText("⚠️ Classifier module not available")
            return

        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS

        if not os.path.exists(weights_path):
            self.recog_status.setText(
                f"⚠️ No weights found at '{weights_path}'\n"
                "Train the classifier first via:\n"
                "  python app/train_classifier.py"
            )
            self.classifier_loaded = False
            return

        try:
            self.classifier = create_classifier(weights_path=weights_path)
            self.classifier_loaded = True
            num_species = self.classifier.num_species
            self.recog_status.setText(
                f"✅ Classifier loaded ({num_species} species)\n"
                f"Weights: {os.path.basename(weights_path)}"
            )
            self.recog_status.setStyleSheet("color: #27ae60; font-weight: bold;")
            self.btn_batch_classify.setEnabled(True)
            print(f"🔬 Species classifier ready: {num_species} species")
        except Exception as e:
            self.recog_status.setText(f"❌ Failed to load classifier: {str(e)}")
            self.classifier_loaded = False

    # ── UI Setup ──────────────────────────────────────────────────────────

    def setup_ui(self):
        main_layout = QVBoxLayout()

        # ── Top: Inner tab widget (Live / Batch / Analytics) ──
        self.inner_tabs = QTabWidget()

        # --- Tab A: Live Recognition ---
        live_tab = QWidget()
        live_layout = QVBoxLayout()

        # Camera feed + prediction overlay
        self.live_feed = QLabel("Live Camera Feed\n(Awaiting frames...)")
        self.live_feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_feed.setStyleSheet("background: black; border: 2px solid #34495e; color: white;")
        self.live_feed.setMinimumSize(640, 480)
        live_layout.addWidget(self.live_feed, stretch=2)

        # Prediction display
        pred_group = QGroupBox("Live Prediction")
        pred_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #7f8c8d; "
            "border-radius: 6px; margin-top: 8px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; }"
        )
        pred_layout = QVBoxLayout()

        self.pred_species_label = QLabel("Species: --")
        self.pred_species_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #2c3e50;")
        self.pred_species_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pred_layout.addWidget(self.pred_species_label)

        self.pred_confidence_label = QLabel("Confidence: --")
        self.pred_confidence_label.setStyleSheet("font-size: 18px; color: #7f8c8d;")
        self.pred_confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pred_layout.addWidget(self.pred_confidence_label)

        # Top-3 predictions
        self.top3_label = QLabel("")
        self.top3_label.setStyleSheet("font-size: 13px; color: #95a5a6;")
        self.top3_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pred_layout.addWidget(self.top3_label)

        # Controls row
        ctrl_row = QHBoxLayout()
        self.chk_auto_classify = QCheckBox("Auto-Classify Live Feed")
        self.chk_auto_classify.setChecked(True)
        ctrl_row.addWidget(self.chk_auto_classify)

        ctrl_row.addWidget(QLabel("Interval (s):"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 30)
        self.interval_spin.setValue(2)
        self.interval_spin.setSuffix("s")
        ctrl_row.addWidget(self.interval_spin)

        self.btn_classify_now = QPushButton("🔍 Classify Now")
        self.btn_classify_now.clicked.connect(self.classify_current_frame)
        self.btn_classify_now.setStyleSheet(
            "background-color: #2980b9; color: white; font-weight: bold; padding: 6px 16px;"
        )
        ctrl_row.addWidget(self.btn_classify_now)
        ctrl_row.addStretch()
        pred_layout.addLayout(ctrl_row)

        pred_group.setLayout(pred_layout)
        live_layout.addWidget(pred_group)

        live_tab.setLayout(live_layout)
        self.inner_tabs.addTab(live_tab, "🎯 Live Recognition")

        # --- Tab B: Batch Classification ---
        batch_tab = QWidget()
        batch_layout = QVBoxLayout()

        batch_controls = QHBoxLayout()
        self.btn_batch_classify = QPushButton("📦 Run Batch Classification")
        self.btn_batch_classify.setEnabled(False)
        self.btn_batch_classify.setStyleSheet(
            "background-color: #8e44ad; color: white; font-weight: bold; padding: 8px 20px;"
        )
        self.btn_batch_classify.clicked.connect(self.run_batch_classification)
        batch_controls.addWidget(self.btn_batch_classify)

        self.btn_export_csv = QPushButton("📤 Export to CSV")
        self.btn_export_csv.clicked.connect(self.export_classifications_csv)
        batch_controls.addWidget(self.btn_export_csv)

        batch_controls.addStretch()
        batch_layout.addLayout(batch_controls)

        self.batch_progress = QProgressBar()
        self.batch_progress.setValue(0)
        batch_layout.addWidget(self.batch_progress)

        self.batch_console = QTextEdit()
        self.batch_console.setReadOnly(True)
        self.batch_console.setStyleSheet(
            "background-color: #2c3e50; color: #ecf0f1; "
            "font-family: Consolas; font-size: 11px;"
        )
        batch_layout.addWidget(self.batch_console, stretch=1)

        batch_tab.setLayout(batch_layout)
        self.inner_tabs.addTab(batch_tab, "📦 Batch Classification")

        # --- Tab C: Analytics ---
        self.analytics_widget = AnalyticsWidget()
        self.inner_tabs.addTab(self.analytics_widget, "📊 Analytics")

        main_layout.addWidget(self.inner_tabs, stretch=1)

        # ── Bottom: Status bar ──
        status_layout = QHBoxLayout()
        self.recog_status = QLabel("Initializing...")
        self.recog_status.setStyleSheet("color: #7f8c8d; font-style: italic;")
        status_layout.addWidget(self.recog_status, stretch=1)

        self.btn_reload = QPushButton("🔄 Reload Classifier")
        self.btn_reload.clicked.connect(self.reload_classifier)
        status_layout.addWidget(self.btn_reload)

        main_layout.addLayout(status_layout)

        self.setLayout(main_layout)

    # ── Live Feed Update (called from main.py) ────────────────────────────

    def update_live_feed(self, frame):
        """Called by main.py's camera thread to push frames."""
        self.current_frame = frame

        if not self.live_mode:
            return

        # Display the frame
        pix = self._convert_cv_to_qpixmap(frame, self.live_feed.width(), self.live_feed.height())
        self.live_feed.setPixmap(pix)

        # Auto-classify if enabled
        if self.chk_auto_classify.isChecked() and self.classifier_loaded:
            current_time = time.time()
            interval = self.interval_spin.value()
            if current_time - self.last_classify_time >= interval:
                self.last_classify_time = current_time
                self._run_live_classification(frame)

    def _run_live_classification(self, frame):
        """Run classification on a frame and update the prediction labels."""
        try:
            species, confidence, top3 = self.classifier.predict(frame, top_k=3)

            self.pred_species_label.setText(f"Species: {species}")
            self.pred_confidence_label.setText(f"Confidence: {confidence:.1%}")

            # Color-code confidence
            if confidence >= 0.8:
                self.pred_species_label.setStyleSheet(
                    "font-size: 28px; font-weight: bold; color: #27ae60;"
                )
            elif confidence >= 0.5:
                self.pred_species_label.setStyleSheet(
                    "font-size: 28px; font-weight: bold; color: #f39c12;"
                )
            else:
                self.pred_species_label.setStyleSheet(
                    "font-size: 28px; font-weight: bold; color: #e74c3c;"
                )

            # Top-3 display
            top3_text = " | ".join([f"{s}: {c:.0%}" for s, c in top3])
            self.top3_label.setText(f"Top 3: {top3_text}")

        except Exception as e:
            self.pred_species_label.setText("Classification Error")
            self.pred_confidence_label.setText(str(e))

    def classify_current_frame(self):
        """Manually classify the current frame."""
        if self.current_frame is not None and self.classifier_loaded:
            self._run_live_classification(self.current_frame)

    # ── Batch Classification ──────────────────────────────────────────────

    def run_batch_classification(self):
        """Start the batch classification thread."""
        if not self.classifier_loaded:
            QMessageBox.warning(self, "Classifier Not Loaded",
                                "Please train or load a classifier first.")
            return

        self.btn_batch_classify.setEnabled(False)
        self.batch_console.clear()
        self.batch_progress.setValue(0)

        self.batch_thread = BatchClassificationThread(
            classifier=self.classifier,
            db_path=DB_PATH,
            confidence_threshold=0.3,
        )
        self.batch_thread.log_signal.connect(self.batch_console.append)
        self.batch_thread.progress_signal.connect(self.batch_progress.setValue)
        self.batch_thread.finished_signal.connect(self.on_batch_finished)
        self.batch_thread.start()

    def on_batch_finished(self):
        self.btn_batch_classify.setEnabled(True)
        self.batch_console.append("\n✅ Batch classification complete!")
        # Refresh analytics
        self.analytics_widget.refresh()

    def export_classifications_csv(self):
        """Export all classification results to a CSV file."""
        if not os.path.exists(DB_PATH):
            QMessageBox.warning(self, "No Data", "Database not found.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Classifications CSV", "classifications_export.csv",
            "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.sample_name, c.predicted_species, c.confidence,
                       c.top3_predictions, c.timestamp, s.species_name as actual_species
                FROM classifications c
                LEFT JOIN samples s ON c.sample_name = s.sample_name
                ORDER BY c.timestamp DESC
            """)
            rows = cursor.fetchall()
            conn.close()

            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Sample Name", "Predicted Species", "Confidence",
                    "Top 3", "Timestamp", "Actual Species", "Correct"
                ])
                for row in rows:
                    sample, pred, conf, top3_json, ts, actual = row
                    correct = "YES" if pred == actual else "NO" if actual else "N/A"
                    writer.writerow([sample, pred, f"{conf:.2%}", top3_json, ts, actual, correct])

            QMessageBox.information(self, "Export Complete",
                                    f"Exported {len(rows)} classifications to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    # ── Utilities ─────────────────────────────────────────────────────────

    def reload_classifier(self):
        """Reload the classifier (e.g., after training new weights)."""
        self.load_classifier()

    def _convert_cv_to_qpixmap(self, frame, label_width, label_height):
        if len(frame.shape) == 2:
            h, w = frame.shape
            q_img = QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, ch = frame.shape
            q_img = QImage(frame.data, w, h, w * ch, QImage.Format.Format_BGR888)
        return QPixmap.fromImage(q_img).scaled(
            QSize(label_width, label_height), Qt.AspectRatioMode.KeepAspectRatio
        )
