# app/models.py

import torch
import torch.nn as nn

# --- 1. Baseline: Simple CNN (Section 3.6.2.1) ---
class SimpleRestorationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.encoder(x)

# --- 2. Custom Model: SRCNN (Section 3.6.2.2) ---
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

# --- Future Advanced Architectures (Sections 3.6.2.3 - 3.6.2.5) ---
# We will integrate the massive architectures for these once your baseline is proven.

class VDSR(nn.Module):
    pass # To be implemented

class SwinIR(nn.Module):
    pass # To be implemented

class RealESRGAN_Wrapper():
    pass # To be implemented (Will use pre-trained external weights)