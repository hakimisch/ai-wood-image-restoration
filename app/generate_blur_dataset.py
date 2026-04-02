# generate_blur_dataset.py
#
# Run this ONCE before training to pre-generate all blurred images.
# It reads every clear image from the database, applies a random blur
# (Gaussian, out-of-focus, or motion) with continuous sigma, saves the
# blurred image to data/blurred/, and writes the blur_path back to the DB.
#
# After this runs, training becomes pure disk I/O — no CPU blur math
# per batch, no worker pickling overhead, maximum GPU utilisation.
#
# Usage:
#   python generate_blur_dataset.py
#
# Re-run any time you want to regenerate with fresh random blurs.
# Each run overwrites the blurred folder and resets blur_path in the DB.
 
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
    k = max(3, int(np.ceil(6 * sigma)) | 1)  # Always odd, min 3
    # cv2.GaussianBlur expects (kW, kH) and sigma
    return cv2.GaussianBlur(img_rgb, (k, k), sigmaX=sigma, sigmaY=sigma)
 
 
def make_out_of_focus_blur(img_rgb, sigma):
    """Disk/elliptical bokeh blur simulating physical defocus."""
    k = max(3, int(np.ceil(6 * sigma)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    kernel = kernel.astype(np.float32) / kernel.sum()
    # filter2D works on float32 [0,1] or uint8 — works on RGB directly
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

    # 1. Pick two extremes for the focal tilt (e.g., sharp vs heavily out of focus)
    sigma_sharp  = random.uniform(0.1, 1.0)
    sigma_blurry = random.uniform(3.0, 6.0)

    # Focal tilt is a physical lens effect, so we use out-of-focus (disk) blur
    img_sharp  = make_out_of_focus_blur(img_rgb, sigma_sharp).astype(np.float32)
    img_blurry = make_out_of_focus_blur(img_rgb, sigma_blurry).astype(np.float32)

    # 2. Generate a 2D linear gradient mask [0.0 to 1.0]
    mask = np.zeros((h, w, 1), dtype=np.float32)
    is_vertical = random.random() > 0.5

    if is_vertical:
        # Tilt from top-to-bottom
        grad = np.linspace(0, 1, h, dtype=np.float32)
        if random.random() > 0.5: grad = grad[::-1] # Flip direction
        mask[:, :, 0] = np.tile(grad[:, None], (1, w))
    else:
        # Tilt from left-to-right
        grad = np.linspace(0, 1, w, dtype=np.float32)
        if random.random() > 0.5: grad = grad[::-1] # Flip direction
        mask[:, :, 0] = np.tile(grad[None, :], (h, 1))

    # 3. Blend the two images using the gradient mask
    blended = (img_sharp * mask) + (img_blurry * (1.0 - mask))
    return np.clip(blended, 0, 255).astype(np.uint8)
 
 
def apply_random_blur(img_rgb):
    """Pick a random blur type and strength, return blurred RGB image."""
    rand_val = random.random()
    
    # 1. THE >900 VOL FIX (10% Chance: Identity Mapping)
    # Teaches the AI to leave already-sharp images alone
    if rand_val < 0.10:
        return img_rgb
        
    # 2. THE TILTED STAGE FIX (20% Chance: Space-Variant Blur)
    # Teaches the AI to fix uneven wood surfaces and focal plane drift
    if rand_val < 0.30:
        return make_space_variant_blur(img_rgb)

    # 3. THE <300 VOL FIX (70% Chance: Standard Uniform Blur)
    # Widened sigma to 6.0 to handle extreme microscopic defocus
    blur_type = random.choice(["gaussian", "out_of_focus", "motion"])
    sigma     = random.uniform(0.1, 6.0)   

    if blur_type == "gaussian":
        return make_gaussian_blur(img_rgb, sigma)
    elif blur_type == "out_of_focus":
        return make_out_of_focus_blur(img_rgb, sigma)
    else:
        return make_motion_blur(img_rgb)
 
# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
 
conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
 
# Ensure blur_path column exists (it may already from the original schema)
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
print("NOTE: This overwrites any existing blur_path values (including old fixed Gaussian blurs")
print(f"      saved by the acquisition tab). New images go to '{BLUR_OUT_DIR}/'.")
 
for i, (sample_id, clear_path) in enumerate(rows):
    # Read clear image
    img_bgr = cv2.imread(clear_path)
    if img_bgr is None:
        print(f"  ⚠️  Could not read {clear_path}, skipping.")
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
 
    # Apply random blur
    blurred_rgb = apply_random_blur(img_rgb)
    blurred_rgb = np.clip(blurred_rgb, 0, 255).astype(np.uint8)
 
    # Build output path: data/blurred/<original_filename>
    filename  = os.path.basename(clear_path)
    blur_path = os.path.join(BLUR_OUT_DIR, filename).replace('\\', '/')
 
    # Save as BGR (OpenCV convention)
    blurred_bgr = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(blur_path, blurred_bgr)
 
    # Write path back to DB
    cursor.execute("UPDATE samples SET blur_path = ? WHERE id = ?", (blur_path, sample_id))
 
    # Progress report every 500 images
    if (i + 1) % 500 == 0 or (i + 1) == total:
        print(f"  [{i+1}/{total}] Done.")
 
conn.commit()
conn.close()
print(f"\n✅ Blur generation complete. {total} blurred images saved to '{BLUR_OUT_DIR}/'.")
print("You can now run training — epochs will use pre-generated blur pairs with no CPU overhead.")