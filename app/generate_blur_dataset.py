# generate_blur_dataset.py
#
# Run this ONCE before training to pre-generate all blurred images.
# It reads every clear image from the database, applies a random blur
# (Gaussian, out-of-focus, or motion) with continuous sigma, saves the
# blurred image to data/blurred/, and writes the blur_path back to the DB.

import os
import cv2
import random
import sqlite3
import numpy as np

DB_PATH      = 'data/database.db'
BLUR_OUT_DIR = 'data/blurred'

os.makedirs(BLUR_OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Blur kernel helpers
# ---------------------------------------------------------------------------

def make_gaussian_blur(img_rgb, sigma):
    """Standard Gaussian blur via torchvision-compatible sigma."""
    k = max(3, int(np.ceil(6 * sigma)) | 1)  
    return cv2.GaussianBlur(img_rgb, (k, k), sigmaX=sigma, sigmaY=sigma)

def make_out_of_focus_blur(img_rgb, sigma):
    """Disk/elliptical bokeh blur simulating physical defocus."""
    k = max(3, int(np.ceil(6 * sigma)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    kernel = kernel.astype(np.float32) / kernel.sum()
    return cv2.filter2D(img_rgb, -1, kernel)

def make_motion_blur(img_rgb):
    """Linear motion blur at a random angle and streak length."""
    length = random.randint(5, 21)
    angle  = random.uniform(0, 360)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    M = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    return cv2.filter2D(img_rgb, -1, kernel)

def make_space_variant_blur(img_rgb):
    """Simulates a tilted microscope stage (focal plane gradient) via alpha blending."""
    h, w = img_rgb.shape[:2]
    sigma_sharp  = random.uniform(0.1, 1.0)
    sigma_blurry = random.uniform(3.0, 6.0)

    img_sharp  = make_out_of_focus_blur(img_rgb, sigma_sharp).astype(np.float32)
    img_blurry = make_out_of_focus_blur(img_rgb, sigma_blurry).astype(np.float32)

    mask = np.zeros((h, w, 1), dtype=np.float32)
    is_vertical = random.random() > 0.5

    if is_vertical:
        grad = np.linspace(0, 1, h, dtype=np.float32)
        if random.random() > 0.5: grad = grad[::-1] 
        mask[:, :, 0] = np.tile(grad[:, None], (1, w))
    else:
        grad = np.linspace(0, 1, w, dtype=np.float32)
        if random.random() > 0.5: grad = grad[::-1] 
        mask[:, :, 0] = np.tile(grad[None, :], (h, 1))

    blended = (img_sharp * mask) + (img_blurry * (1.0 - mask))
    return np.clip(blended, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# Camera Degradation Helpers
# ---------------------------------------------------------------------------

def add_sensor_noise(img_rgb, sigma=None):
    """Gaussian sensor noise matching typical USB microscope cameras."""
    if sigma is None:
        sigma = random.uniform(1.0, 6.0) # Tuned down to prevent information death
    noise = np.random.normal(0, sigma, img_rgb.shape).astype(np.float32)
    return np.clip(img_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def add_camera_banding(img_rgb):
    """Simulates LED PWM flicker and rolling shutter waves found in live microscope feeds."""
    h, w = img_rgb.shape[:2]
    
    # Create coordinates for a 2D wave
    y = np.arange(h).reshape(-1, 1)
    x = np.arange(w).reshape(1, -1)
    
    # Randomize wave frequency (width) and tilt (diagonal shift)
    freq_y = random.uniform(0.02, 0.08)
    freq_x = random.uniform(-0.02, 0.02) 
    
    # Amplitude controls how dark/light the bands get. Keep it faint (like the real camera)
    amplitude = random.uniform(3.0, 12.0) 
    
    # Generate the sine wave mask
    wave = (np.sin(y * freq_y + x * freq_x) * amplitude).astype(np.float32)
    wave_3d = np.repeat(wave[:, :, np.newaxis], 3, axis=2)
    
    # Apply the banding to the image
    banded = np.clip(img_rgb.astype(np.float32) + wave_3d, 0, 255).astype(np.uint8)
    return banded

def simulate_camera_isp(img_rgb):
    """Simulates the camera's internal Auto-Exposure and Gamma correction."""
    # Gamma < 1.0 darkens shadows/increases contrast, Gamma > 1.0 washes out
    gamma = random.uniform(0.7, 1.2)
    
    # Build a lookup table for fast Gamma correction
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(img_rgb, table)

def add_lighting_variance(img_rgb):
    """LED flicker and varied sample exposure."""
    alpha = random.uniform(0.8, 1.2)   
    beta  = random.uniform(-20, 20)    
    return cv2.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)

def add_vignetting(img_rgb):
    """Darker edges with a randomized, off-center focal point."""
    rows, cols = img_rgb.shape[:2]
    
    # Shift the center of the light by up to 30% in any direction
    center_x = (cols / 2) + random.uniform(-cols * 0.3, cols * 0.3)
    center_y = (rows / 2) + random.uniform(-rows * 0.3, rows * 0.3)
    
    # Increase the kernel size slightly to soften the edge of the light pool
    kx = cv2.getGaussianKernel(cols, cols / 1.5)
    ky = cv2.getGaussianKernel(rows, rows / 1.5)
    
    # Shift the kernel to the new center
    kx = np.roll(kx, int(center_x - cols/2))
    ky = np.roll(ky, int(center_y - rows/2))
    
    mask = (ky * kx.T)
    mask = mask / mask.max()
    mask = np.expand_dims(mask, axis=2)
    return np.clip(img_rgb * mask, 0, 255).astype(np.uint8)

def add_jpeg_compression(image_rgb):
    """Simulates the compression artifacts from a USB webcam stream."""
    quality = random.randint(50, 95)
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
    decoded_bgr = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------------------------------
# The Master Degradation Pipeline
# ---------------------------------------------------------------------------

def apply_random_blur(img_rgb):
    """Full physically-realistic degradation pipeline."""
    rand_val = random.random()

    # ── Logic Gate: Prevent Contrast Stacking (The "Claude Fix") ──────────
    # Decide HERE if we are doing physical lighting OR digital ISP, but never both.
    apply_physical_light = random.random() > 0.5
    apply_digital_isp    = not apply_physical_light and random.random() > 0.5
    # ──────────────────────────────────────────────────────────────────────

    # ── Stage 1: Lens & illumination ──────────────────────────────────────
    if apply_physical_light:
        img_rgb = add_lighting_variance(img_rgb)
        
    if random.random() > 0.5:
        img_rgb = add_vignetting(img_rgb)

   # ── Stage 2: Optical blur (Compound Simulation) ───────────────────────
    if rand_val >= 0.10: # 90% chance to get blurred
        
        # 1. GUARANTEED Base Focus
        if random.random() < 0.25:
            # 25% chance of space-variant (tilted stage)
            img_rgb = make_space_variant_blur(img_rgb)
        else:
            # 75% chance of standard uniform blur
            sigma = random.uniform(0.5, 5.0) 
            
            if random.random() < 0.5:
                img_rgb = make_out_of_focus_blur(img_rgb, sigma)
            else:
                img_rgb = make_gaussian_blur(img_rgb, sigma)
                
        # 2. Additive Motion/Vibration (30% chance to stack on top of the base blur)
        if random.random() < 0.30:
            img_rgb = make_motion_blur(img_rgb)

    # ── Stage 3: Sensor & capture ─────────────────────────────────────────
    # Inject the LED banding simulation ~50% of the time
    if random.random() > 0.50:
        img_rgb = add_camera_banding(img_rgb)
        
    img_rgb = add_sensor_noise(img_rgb)         
    
    # This now safely checks the variable we defined at the very top
    if apply_digital_isp:
        img_rgb = simulate_camera_isp(img_rgb)
        
    if random.random() > 0.30:                  
        img_rgb = add_jpeg_compression(img_rgb)
        
    return img_rgb
 
# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
 
conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
 
cursor.execute("PRAGMA table_info(samples)")
columns = [row[1] for row in cursor.fetchall()]
if 'blur_path' not in columns:
    cursor.execute("ALTER TABLE samples ADD COLUMN blur_path TEXT")
    conn.commit()
    print("Added blur_path column to samples table.")
 
cursor.execute("SELECT id, clear_path FROM samples")
rows = cursor.fetchall()
total = len(rows)
print(f"Generating blurred images for {total} samples...")
 
for i, (sample_id, clear_path) in enumerate(rows):
    
    # 1. Read clean image
    img_bgr = cv2.imread(clear_path)
    if img_bgr is None:
        print(f"  ⚠️  Could not read {clear_path}, skipping.")
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Apply Master Pipeline (Lighting -> Blur -> Noise -> Compression)
    blurred_rgb = apply_random_blur(img_rgb)
    blurred_rgb = np.clip(blurred_rgb, 0, 255).astype(np.uint8)
 
    # 3. Save as BGR (OpenCV convention)
    filename  = os.path.basename(clear_path)
    blur_path = os.path.join(BLUR_OUT_DIR, filename).replace('\\', '/')
    blurred_bgr = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(blur_path, blurred_bgr)
 
    # 4. Write path back to DB
    cursor.execute("UPDATE samples SET blur_path = ? WHERE id = ?", (blur_path, sample_id))
 
    # Progress report
    if (i + 1) % 500 == 0 or (i + 1) == total:
        print(f"  [{i+1}/{total}] Done.")
 
conn.commit()
conn.close()
print(f"\n✅ Blur generation complete. {total} blurred images saved to '{BLUR_OUT_DIR}/'.")