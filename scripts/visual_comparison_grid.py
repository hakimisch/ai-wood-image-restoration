"""
scripts/visual_comparison_grid.py

Generate side-by-side comparison: Blurry | VDSR | SwinIR | ESRGAN | Ground Truth
Uses the correctly restored images from outputs/fft_analysis/
"""
import cv2, os, glob, numpy as np
from matplotlib import pyplot as plt

out_dir = 'outputs/fft_analysis'
clear_dir = f'{out_dir}/clear'
files = sorted(glob.glob(f'{clear_dir}/*.jpg'))

# Use a mid-range sample that shows clear differences
# Find images with moderate blur (VoL between 300-800) where restoration helps
target_file = None
for f in files:
    img = cv2.imread(f)
    blur_img = cv2.imread(f.replace('/clear/', '/blurry/'))
    if img is not None and blur_img is not None:
        clear_vol = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        blur_vol = cv2.Laplacian(cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if 300 < blur_vol < 800:
            target_file = os.path.basename(f)
            print(f'Selected: {target_file} (blur VoL={blur_vol:.0f}, clear VoL={clear_vol:.0f})')
            break

if not target_file:
    target_file = os.path.basename(files[0])
    print(f'Using first file: {target_file}')

# Load images
labels = ['Blurry Input', 'VDSR', 'SwinIR', 'Real-ESRGAN', 'Ground Truth']
keys = ['blurry', 'VDSR', 'SwinIR', 'Real-ESRGAN', 'clear']

images = []
for key in keys:
    path = f'{out_dir}/{key}/{target_file}'
    img = cv2.imread(path)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
    else:
        print(f'Warning: {path} not found')
        images.append(np.zeros((100, 100, 3), dtype=np.uint8))

# Create side-by-side montage
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, img, label in zip(axes, images, labels):
    ax.imshow(img)
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
os.makedirs('research', exist_ok=True)
out_path = 'research/visual_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.close()

# Also save a 2-row version with zoomed crops
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
h, w = images[0].shape[:2]
cy, cx = h // 2, w // 2
crop_sz = 100

for i, (img, label) in enumerate(zip(images, labels)):
    # Full image
    axes[0, i].imshow(img)
    axes[0, i].set_title(label, fontsize=11, fontweight='bold')
    axes[0, i].axis('off')
    # Draw rectangle showing crop region
    rect = plt.Rectangle((cx-crop_sz, cy-crop_sz), crop_sz*2, crop_sz*2,
                          fill=False, edgecolor='red', linewidth=1.5)
    axes[0, i].add_patch(rect)
    
    # Zoomed crop
    crop = img[cy-crop_sz:cy+crop_sz, cx-crop_sz:cx+crop_sz]
    axes[1, i].imshow(crop)
    axes[1, i].set_title(f'Detail ({crop_sz*2}x{crop_sz*2}px)', fontsize=9)
    axes[1, i].axis('off')

plt.tight_layout()
out_path = 'research/visual_comparison_zoomed.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.close()

print(f'\nImage dimensions: {images[0].shape[1]}x{images[0].shape[0]}')
for label, img in zip(labels, images):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    vol = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f'  {label:20s}: VoL={vol:7.0f}')
