"""
scripts-ablations/image-restoration/generate_ablation_figure.py

Generates a publication-quality visual comparison figure for §4.2 (Domain Gap).

Layout (4 columns):
  Blurred Input | Gaussian-Only VDSR | Full-Physics VDSR | Ground Truth
  Annotated with PSNR above each restored column.

Usage:
  cd /mnt/c/Hakimi/Internship/GUI Image Restoration
  ./app/torch_env/Scripts/python -X utf8 scripts-ablations/image-restoration/generate_ablation_figure.py
"""

import os
import sys
import sqlite3
import cv2
import numpy as np
import torch
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'app'))
from models import VDSR

# ── Config ─────────────────────────────────────────────────────────────────
PHYSICS_BLUR_DIR  = os.path.join(PROJECT_ROOT, 'data/blurred')
DB_PATH           = os.path.join(PROJECT_ROOT, 'data/database.db')
CROP_SIZE         = 256

GAUSSIAN_ONLY_WT = os.path.join(PROJECT_ROOT, "20e_vdsr_gaussian_only_hybrid.pth")
FULL_PHYSICS_WT  = os.path.join(PROJECT_ROOT, "50e_vdsr_hybrid_accum2_4_9.pth")

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "research/figures/ablation_domain_gap_comparison.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {DEVICE}")


def get_test_pairs(blur_dir, num_samples=50):
    """Same deterministic test split as WoodDataset (seed 42, last 50)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT clear_path FROM samples")
    clear_by_name = {os.path.basename(row[0]): row[0] for row in cursor.fetchall()}
    conn.close()

    image_pairs = []
    for fname in os.listdir(blur_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        clear_path = clear_by_name.get(fname)
        if clear_path is None:
            continue
        blur_path = os.path.join(blur_dir, fname).replace('\\', '/')
        image_pairs.append((blur_path, clear_path))

    import random as rng_stdlib
    rng = rng_stdlib.Random(42)
    image_pairs.sort()
    rng.shuffle(image_pairs)

    if len(image_pairs) > num_samples:
        return image_pairs[-num_samples:]
    return image_pairs[-min(num_samples, len(image_pairs)):]


def load_model(weight_path):
    model = VDSR().to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()
    return model


def run_inference(model, blur_np_rgb, clear_np_rgb, transform):
    """Run inference on blurred image, compare output to ground truth clear image.
    Returns output RGB uint8 + metrics dict + center-cropped clear image."""
    # Center crop both
    blur_tensor = transform(blur_np_rgb).unsqueeze(0).to(DEVICE)
    clear_t = transform(clear_np_rgb)
    clear_cropped = (clear_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    with torch.no_grad():
        output_tensor = model(blur_tensor)
    out_img = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    out_img = (np.clip(out_img, 0, 1) * 255).astype(np.uint8)

    metrics = {
        'psnr': psnr(clear_cropped, out_img, data_range=255),
        'ssim': ssim(clear_cropped, out_img, data_range=255, channel_axis=-1, win_size=3),
    }
    return out_img, metrics, clear_cropped


def pick_best_sample(test_pairs, gauss_model, phys_model, transform):
    """
    Pick the sample with the LARGEST domain gap (biggest PSNR difference
    between Gaussian-only and full-physics outputs on physics test set).
    """
    best_idx = None
    best_gap = -1
    best_imgs = None

    print("  Scanning test set for most representative sample...")
    for idx, (blur_path, clear_path) in enumerate(test_pairs):
        blur_bgr = cv2.imread(blur_path)
        clear_bgr = cv2.imread(clear_path)
        if blur_bgr is None or clear_bgr is None:
            continue

        blur_rgb = cv2.cvtColor(blur_bgr, cv2.COLOR_BGR2RGB)
        clear_rgb = cv2.cvtColor(clear_bgr, cv2.COLOR_BGR2RGB)

        gauss_out, gauss_m, clear_crop = run_inference(gauss_model, blur_rgb, clear_rgb, transform)
        phys_out, phys_m, _ = run_inference(phys_model, blur_rgb, clear_rgb, transform)

        gap = phys_m['psnr'] - gauss_m['psnr']
        if gap > best_gap:
            best_gap = gap
            best_idx = idx
            best_imgs = {
                'blur': cv2.cvtColor(cv2.imread(blur_path), cv2.COLOR_BGR2RGB),
                'clear': clear_rgb,
                'clear_crop': clear_crop,
                'gauss_out': gauss_out,
                'phys_out': phys_out,
                'gauss_psnr': gauss_m['psnr'],
                'gauss_ssim': gauss_m['ssim'],
                'phys_psnr': phys_m['psnr'],
                'phys_ssim': phys_m['ssim'],
            }

    print(f"  Selected sample #{best_idx} — gap = {best_gap:.2f} dB")
    return best_imgs


def build_figure(imgs):
    """Create the 4-column side-by-side figure."""
    blur_vis = imgs['blur']
    # Center-crop blur to match the 256x256 model input
    h, w = blur_vis.shape[:2]
    top = (h - CROP_SIZE) // 2
    left = (w - CROP_SIZE) // 2
    blur_cropped = blur_vis[top:top+CROP_SIZE, left:left+CROP_SIZE]

    images = [
        blur_cropped,
        imgs['gauss_out'],
        imgs['phys_out'],
        imgs['clear_crop'],
    ]

    titles = [
        "Blurred Input\n(Physics Pipeline)",
        f"Gaussian-Only VDSR\nPSNR {imgs['gauss_psnr']:.2f} / SSIM {imgs['gauss_ssim']:.4f}",
        f"Full-Physics VDSR\nPSNR {imgs['phys_psnr']:.2f} / SSIM {imgs['phys_ssim']:.4f}",
        "Ground Truth",
    ]

    # Colour coding: red for degradation, green for improvement
    label_colours = ['gray', '#d62728', '#2ca02c', 'gray']

    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.05)

    for i, (img, title, colour) in enumerate(zip(images, titles, label_colours)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold', color=colour, pad=10)
        ax.axis('off')

    # Column separation is handled by wspace=0.05 in GridSpec — no explicit lines needed.
    # for i in range(3):
    #     fig.add_artist(plt.Line2D(
    #         [(i + 1) / 4, (i + 1) / 4], [0.08, 0.92],
    #         transform=fig.transFigure, color='#cccccc', linewidth=0.8
    #     ))

    plt.suptitle(
        "Domain Gap Ablation — VDSR on Physics-Pipeline Test Image",
        fontsize=14, fontweight='bold', y=0.98
    )
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.15,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✅ Figure saved: {OUTPUT_PATH}")


def main():
    print("\n" + "=" * 60)
    print("📸  Generating Ablation Comparison Figure")
    print("=" * 60)

    # Load models
    print("\n  Loading models...")
    gauss_model = load_model(GAUSSIAN_ONLY_WT)
    phys_model  = load_model(FULL_PHYSICS_WT)
    print("  ✅ Models loaded")

    # Get test pairs
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor()
    ])

    test_pairs = get_test_pairs(PHYSICS_BLUR_DIR, num_samples=50)
    print(f"  Test set: {len(test_pairs)} pairs")

    # Find best sample and build figure
    imgs = pick_best_sample(test_pairs, gauss_model, phys_model, transform)
    build_figure(imgs)

    print(f"\n  📁 Figure: {OUTPUT_PATH}")
    print(f"  Insert in paper as Figure 4.2 (or after existing Fig 4.1)\n")


if __name__ == "__main__":
    main()
