# scripts/evaluate_baselines.py
#
# Classical (non-DL) baseline evaluation: Wiener Filter & Richardson-Lucy.
#
# Benchmarks the 50-image holdout test set through standard deconvolution
# algorithms to provide absolute bottom-floor baselines for the thesis
# results tables. Computes PSNR, SSIM, and LPIPS against ground truth.
#
# Usage:
#   python scripts/evaluate_baselines.py
#   python scripts/evaluate_baselines.py --psf_sigma 2.5 --num_images 50

import os
import sys
import argparse
import sqlite3
import cv2
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Optional imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.restoration import wiener, richardson_lucy
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("❌ scikit-image not installed. Run: pip install scikit-image")
    sys.exit(1)

try:
    import lpips
    import torch
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️  lpips not installed. LPIPS metric will be skipped.")

DB_PATH = 'data/database.db'


def estimate_psf(kernel_size=15, sigma=2.5):
    """Generate a 2D Gaussian PSF matching the acquisition-time blur kernel.

    Args:
        kernel_size: Size of the square kernel (default 15, matching main.py).
        sigma: Standard deviation of the Gaussian (default 2.5).

    Returns:
        psf: 2D float32 array, sum-normalized to 1.0.
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    center = k // 2
    psf = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            psf[i, j] = np.exp(-((i - center)**2 + (j - center)**2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf


def evaluate_baselines(psf_sigma=2.5, num_images=50):
    """Run Wiener filter and RL deconvolution on the test set.

    Args:
        psf_sigma: Sigma for the estimated Gaussian PSF.
        num_images: Number of test images to evaluate (default 50).
    """
    print("=" * 60)
    print("Classical Baseline Evaluation")
    print("=" * 60)

    # Setup LPIPS if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        print(f"  LPIPS model loaded on {device}")
    else:
        print("  LPIPS disabled.")

    # Generate PSF
    psf = estimate_psf(sigma=psf_sigma)
    print(f"  PSF: {psf.shape[0]}x{psf.shape[0]}, sigma={psf_sigma}")

    # Fetch test pairs from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Use the same deterministic approach as WoodDataset: last 50 samples
    cursor.execute("SELECT COUNT(*) FROM samples")
    total = cursor.fetchone()[0]
    offset = max(0, total - num_images)
    cursor.execute(
        "SELECT blur_path, clear_path FROM samples ORDER BY id LIMIT ? OFFSET ?",
        (num_images, offset)
    )
    test_pairs = cursor.fetchall()
    conn.close()
    print(f"  Test set: {len(test_pairs)} image pairs (offset={offset})")
    print()

    # Accumulators
    wiener_psnr_list  = []
    wiener_ssim_list  = []
    wiener_lpips_list = []
    rl_psnr_list      = []
    rl_ssim_list      = []
    rl_lpips_list     = []
    skipped = 0

    for idx, (blur_path, clear_path) in enumerate(test_pairs):
        if not os.path.exists(blur_path) or not os.path.exists(clear_path):
            skipped += 1
            continue

        # Read images
        blur_bgr  = cv2.imread(blur_path)
        clear_bgr = cv2.imread(clear_path)
        if blur_bgr is None or clear_bgr is None:
            skipped += 1
            continue

        # Convert to RGB for metric computation
        blur_rgb  = cv2.cvtColor(blur_bgr, cv2.COLOR_BGR2RGB)
        clear_rgb = cv2.cvtColor(clear_bgr, cv2.COLOR_BGR2RGB)

        # Resize to 256x256 for consistent evaluation
        blur_rgb  = cv2.resize(blur_rgb, (256, 256))
        clear_rgb = cv2.resize(clear_rgb, (256, 256))
        blur_float  = blur_rgb.astype(np.float64)
        clear_float = clear_rgb.astype(np.float64)

        # ── Wiener Filter ──────────────────────────────────────────────
        # noise_var estimated from the Laplacian of the blurred image
        laplacian = cv2.Laplacian(blur_rgb, cv2.CV_64F)
        noise_var = np.var(laplacian) * 0.01  # heuristic scaling
        # Apply Wiener filter channel-wise
        wiener_result = np.zeros_like(blur_float)
        for c in range(3):
            wiener_result[..., c] = wiener(blur_float[..., c], psf, noise_var, clip=True)
        wiener_result = np.clip(wiener_result, 0, 255).astype(np.uint8)

        # ── Richardson-Lucy ────────────────────────────────────────────
        rl_result = np.zeros_like(blur_float)
        for c in range(3):
            rl_result[..., c] = richardson_lucy(blur_float[..., c], psf, num_iter=30, clip=True)
        rl_result = np.clip(rl_result, 0, 255).astype(np.uint8)

        # ── Metrics ────────────────────────────────────────────────────
        # PSNR / SSIM
        wiener_psnr_list.append(psnr(clear_rgb, wiener_result, data_range=255))
        wiener_ssim_list.append(ssim(clear_rgb, wiener_result, data_range=255,
                                       channel_axis=-1, win_size=3))
        rl_psnr_list.append(psnr(clear_rgb, rl_result, data_range=255))
        rl_ssim_list.append(ssim(clear_rgb, rl_result, data_range=255,
                                  channel_axis=-1, win_size=3))

        # LPIPS
        if LPIPS_AVAILABLE and lpips_fn is not None:
            def to_tensor(img_np):
                t = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
                return t
            wiener_lpips_list.append(lpips_fn(to_tensor(wiener_result), to_tensor(clear_rgb)).item())
            rl_lpips_list.append(lpips_fn(to_tensor(rl_result), to_tensor(clear_rgb)).item())

        # Progress
        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_pairs):
            print(f"  Processed {idx + 1}/{len(test_pairs)} images...")

    # ── Final Results ──────────────────────────────────────────────────
    n_valid = len(wiener_psnr_list)
    print()
    print("=" * 60)
    print("📊  CLASSICAL BASELINE RESULTS")
    print("=" * 60)
    print(f"  Valid samples: {n_valid}  (skipped: {skipped})")
    print()

    headers = ["Method", "PSNR (dB)", "SSIM", "LPIPS"]
    col_w = [18, 12, 10, 10]

    def fmt_row(method, psnr_list, ssim_list, lpips_list):
        p = np.mean(psnr_list)
        s = np.mean(ssim_list)
        l = np.mean(lpips_list) if lpips_list else 0.0
        lpips_str = f"{l:.4f}" if LPIPS_AVAILABLE else "N/A"
        return f"  {method:<{col_w[0]}}{p:>{col_w[1]}.2f}{s:>{col_w[2]}.4f}{lpips_str:>{col_w[3]}}"

    print(f"  {'Method':<{col_w[0]}}{'PSNR (dB)':>{col_w[1]}}{'SSIM':>{col_w[2]}}{'LPIPS':>{col_w[3]}}")
    print(f"  {'-'*sum(col_w)}")
    print(fmt_row("Wiener Filter", wiener_psnr_list, wiener_ssim_list, wiener_lpips_list))
    print(fmt_row("Richardson-Lucy", rl_psnr_list, rl_ssim_list, rl_lpips_list))
    print(f"  {'-'*sum(col_w)}")

    # Comparison with deep learning (informational — user should fill from model_metrics table)
    print()
    print("  NOTE: Compare the above with DL results from the model_metrics table.")
    print("  Example reference values (from model_metrics):")
    print("    SRCNN 50ep MSE:         PSNR=19.14, SSIM=0.421")
    print("    VDSR 50ep MSE:          PSNR=18.48, SSIM=0.436")
    print("    SwinIR 50ep Hybrid:     PSNR=24.44, SSIM=0.568")
    print("    Real-ESRGAN 50ep Hybrid: PSNR=20.57, SSIM=0.468")
    print()

    # Save to CSV for thesis tables
    csv_path = "scripts/baseline_results.csv"
    with open(csv_path, 'w') as f:
        f.write("method,psnr,ssim,lpips\n")
        wiener_lpips_mean = np.mean(wiener_lpips_list) if LPIPS_AVAILABLE else 0.0
        rl_lpips_mean     = np.mean(rl_lpips_list)     if LPIPS_AVAILABLE else 0.0
        f.write(f"Wiener,{np.mean(wiener_psnr_list):.2f},{np.mean(wiener_ssim_list):.4f},{wiener_lpips_mean:.4f}\n")
        f.write(f"Richardson-Lucy,{np.mean(rl_psnr_list):.2f},{np.mean(rl_ssim_list):.4f},{rl_lpips_mean:.4f}\n")
    print(f"  ✅ Results saved to {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Classical baseline evaluation (Wiener & RL deconvolution)."
    )
    parser.add_argument('--psf_sigma', type=float, default=2.5,
                        help='Sigma of the estimated Gaussian PSF (default: 2.5).')
    parser.add_argument('--num_images', type=int, default=50,
                        help='Number of test images to evaluate (default: 50).')
    args = parser.parse_args()
    evaluate_baselines(psf_sigma=args.psf_sigma, num_images=args.num_images)
