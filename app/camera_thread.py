# app/camera_thread.py

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.camera_index = 0
        self.running = True

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW) # Use DSHOW for better Windows stability
        while self.running:
            ret, self.frame = self.cap.read()
            if ret and self.frame is not None:
                self.change_pixmap_signal.emit(self.frame)
            else:
                # If frame grabbing fails, wait a tiny bit before trying again
                # This prevents the log-spamming loop
                self.msleep(10) 
        self.cap.release()

    def update_index(self, index):
        self.running = False
        self.wait() # Ensure old thread stops
        self.camera_index = index
        self.running = True
        self.start()