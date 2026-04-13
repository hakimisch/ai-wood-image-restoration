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

# --- 4. Custom Model: SwinIR (Vision Transformer) ---
# Note: This is a Lightweight SwinIR configuration optimized for a 6GB GTX 1660 Ti.

def window_partition(x, window_size):
    """Chops the image tensor into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """Stitches the windows back together into a full image."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """Calculates how every pixel relates to every other pixel within a window."""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define a parameter table for relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    """The core building block that shifts the windows to create global context."""
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=None) 
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class BasicLayer(nn.Module):
    """A sequence of Swin Blocks (Residual Swin Transformer Block - RSTB)."""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, H, W):
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        shortcut = x_flat
        for block in self.blocks:
            x_flat = block(x_flat)
        x_out = x_flat.transpose(1, 2).view(b, c, h, w)
        x_out = self.conv(x_out)
        return x + x_out

class SwinIR(nn.Module):
    """The Master Vision Transformer Wrapper."""
    def __init__(self, img_size=128, in_chans=3, embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], window_size=8):
        super().__init__()
        # 1. Shallow Feature Extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2. Deep Feature Extraction (The Transformer blocks)
        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = BasicLayer(dim=embed_dim, input_resolution=(img_size, img_size),
                               depth=depths[i], num_heads=num_heads[i], window_size=window_size)
            self.layers.append(layer)

        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # 3. High-Quality Image Reconstruction
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

    def forward(self, x):
        # Save the original blurry image for the residual skip connection
        shortcut = x 
        
        # Extract features
        x_first = self.conv_first(x)
        res = x_first
        
        # Pass through the Transformer layers
        for layer in self.layers:
            res = layer(res, x.shape[2], x.shape[3])
            
        res = self.conv_after_body(res)
        res = res + x_first
        
        # Reconstruct the final image
        out = self.conv_last(res)
        
        # Add the learned sharp details back onto the blurry input
        return out + shortcut

# --- 5. Advanced Generative Model: Real-ESRGAN (RRDBNet) ---
class ResidualDenseBlock(nn.Module):
    """Dense block where every layer connects to every other layer."""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirical scaling factor of 0.2 to prevent gradient explosion
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    """The master Real-ESRGAN Generator for 1x Restoration (Deblurring)."""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        # NEW: Mathematically neutralize the network to prevent color drift
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Standard Kaiming initialization for LeakyReLU (a=0.2 in ESRGAN)
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # CRITICAL: Force the final layer to output near-zero residuals initially.
        # This guarantees the network starts with a perfect color match to the original image.
        nn.init.normal_(self.conv_last.weight, mean=0.0, std=0.001)
        if self.conv_last.bias is not None:
            nn.init.zeros_(self.conv_last.bias)

    def forward(self, x):
        # Save the original blurry image to anchor the color
        shortcut = x 
        
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        out = self.conv_last(feat)
        
        # Add the generated details back onto the original color
        return out + shortcut