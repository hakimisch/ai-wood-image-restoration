# app/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Baseline: Simple CNN (Section 3.6.2.1) ---
class SimpleRestorationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
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

# --- 3. Custom Model: VDSR (Section 3.6.2.3) ---
class VDSR(nn.Module):
    def __init__(self, num_layers=20): 
        super(VDSR, self).__init__()
        
        # 1. First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        
        # 2. 18 Middle residual layers
        self.conv_residual = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), 
                          nn.ReLU(inplace=True))
            for _ in range(num_layers - 2)
        ])
        
        # 3. Final layer maps back to 3 RGB channels
        self.conv_final = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Manual weight initialization for academic precision
        self._initialize_weights()

    def _initialize_weights(self):
        # Academic standard Kaiming/He initialization for ReLU networks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # For standard conv layers followed by ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # --- THE FINAL VDSR FIX: Correcting final layer initialization ---
        # m.conv_final does NOT have an activation, it is linear.
        # Initializing it with linearity='relu' leads to suboptimal results.
        # We manually re-initialize it here to 'linear' for academic precision.
        nn.init.kaiming_normal_(self.conv_final.weight, mode='fan_out', nonlinearity='linear')
        if self.conv_final.bias is not None:
             nn.init.zeros_(self.conv_final.bias)

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv_residual(x)
        x = self.conv_final(x)
        # VDSR trick: Predict the missing details (x), then add it to the original blurry image (residual)!
        return x + residual

class SwinIR(nn.Module):
    pass # To be implemented

class RealESRGAN_Wrapper():
    pass # To be implemented (Will use pre-trained external weights)