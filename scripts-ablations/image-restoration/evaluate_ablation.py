"""
scripts-ablations/image-restoration/evaluate_ablation.py

VDSR Ablation Evaluation — Fix #6: Side-by-side comparison of Gaussian-only
vs. Full-Physics VDSR on the same physics-pipeline test set.

Purpose:
  Demonstrate the optical domain gap quantitatively. The paper claims (§4.2)
  that Gaussian-only models catastrophically fail on physics-pipeline blurs.
  This script produces the numbers to prove it.

Usage:
  cd /mnt/c/Hakimi/Internship/GUI Image Restoration
  ./app/torch_env/Scripts/python -X utf8 scripts-ablations/image-restoration/evaluate_ablation.py

Models evaluated:
  1. 20e_vdsr_gaussian_only_hybrid.pth  — VDSR trained on Gaussian-only blur
  2. 50e_vdsr_hybrid_accum2_4_9.pth      — VDSR trained on full physics pipeline

Test sets:
  A. Physics-pipeline blurs (data/blurred/) — the REAL test, where Gaussian-only
     should catastrophically fail
  B. Gaussian-only blurs (data/blurred_gaussian/) — sanity check, both should
     perform similarly

Output:
  - Console summary table
  - CSV export: eval_results/ablation_vdsr_domain_gap.csv
"""

import os
import sys
import sqlite3
import csv
import cv2
import numpy as np
import torch
from torchvision import transforms

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# LPIPS (optional)
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️  lpips not installed. LPIPS metric will be skipped.\n")

# Add app/ to path for model imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'app'))
from models import VDSR


# ── Configuration ────────────────────────────────────────────────────────────
PHYSICS_BLUR_DIR   = os.path.join(PROJECT_ROOT, 'data/blurred')          # physics pipeline
GAUSSIAN_BLUR_DIR  = os.path.join(PROJECT_ROOT, 'data/blurred_gaussian') # Gaussian-only
DB_PATH            = os.path.join(PROJECT_ROOT, 'data/database.db')
CROP_SIZE          = 256  # VDSR training crop size

GAUSSIAN_ONLY_WEIGHT = os.path.join(PROJECT_ROOT, "20e_vdsr_gaussian_only_hybrid.pth")
FULL_PHYSICS_WEIGHT  = os.path.join(PROJECT_ROOT, "50e_vdsr_hybrid_accum2_4_9.pth")

OUTPUT_CSV = os.path.join(PROJECT_ROOT, "eval_results", "ablation_vdsr_domain_gap.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {DEVICE}")


def get_test_pairs(blur_dir, num_samples=50):
    """
    Build a list of (blur_path, clear_path) pairs from a blur directory + DB.
    Uses the SAME deterministic seed-42 test split as WoodDataset (last 50).
    """
    # Build filename -> clear_path lookup from DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT clear_path FROM samples")
    clear_by_name = {os.path.basename(row[0]): row[0] for row in cursor.fetchall()}
    conn.close()

    if not os.path.isdir(blur_dir):
        print(f"  ⚠️  Blur directory not found: {blur_dir}")
        return []

    image_pairs = []
    for fname in os.listdir(blur_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        clear_path = clear_by_name.get(fname)
        if clear_path is None:
            continue
        blur_path = os.path.join(blur_dir, fname).replace('\\', '/')
        image_pairs.append((blur_path, clear_path))

    # Same deterministic split as WoodDataset (seed 42, last 50)
    import random as rng_stdlib
    rng = rng_stdlib.Random(42)
    image_pairs.sort()
    rng.shuffle(image_pairs)

    if len(image_pairs) > num_samples:
        image_pairs = image_pairs[-num_samples:]
    else:
        image_pairs = image_pairs[-1:]  # fallback

    return image_pairs


def evaluate_model(weight_path, test_pairs, label="Model"):
    """
    Evaluate a VDSR weight file on a list of (blur_path, clear_path) pairs.
    Returns dict of metrics.
    """
    result = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "valid": 0, "total": len(test_pairs)}

    if not os.path.exists(weight_path):
        print(f"  ❌ Weight file not found: {weight_path}")
        return result

    # Load model
    model = VDSR().to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()
    print(f"  ✅ Loaded: {os.path.basename(weight_path)}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor()
    ])

    # LPIPS setup
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    valid = 0

    with torch.no_grad():
        for idx, (blur_path, clear_path) in enumerate(test_pairs):
            blur_img  = cv2.imread(blur_path)
            clear_img = cv2.imread(clear_path)
            if blur_img is None or clear_img is None:
                continue

            blur_img  = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
            clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)

            # Center-crop ground truth
            clear_tensor = transform(clear_img)
            clear_cropped = (clear_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # Inference
            input_tensor = transform(blur_img).unsqueeze(0).to(DEVICE)
            output_tensor = model(input_tensor)

            output_img = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            output_img = (np.clip(output_img, 0, 1) * 255).astype(np.uint8)

            # Metrics
            total_psnr += psnr(clear_cropped, output_img, data_range=255)
            total_ssim += ssim(clear_cropped, output_img, data_range=255,
                               channel_axis=-1, win_size=3)

            if LPIPS_AVAILABLE and lpips_fn is not None:
                out_t = torch.from_numpy(output_img.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.0
                total_lpips += lpips_fn(out_t, clear_tensor.unsqueeze(0).to(DEVICE)).item()

            valid += 1

    if valid > 0:
        result["psnr"]  = total_psnr / valid
        result["ssim"]  = total_ssim / valid
        result["lpips"] = total_lpips / valid if LPIPS_AVAILABLE else 0.0
        result["valid"] = valid

    return result


def main():
    print("\n" + "=" * 68)
    print("📊  VDSR DOMAIN GAP ABLATION — EVALUATION")
    print("=" * 68)

    # ── Test sets ───────────────────────────────────────────────────────
    physics_test = get_test_pairs(PHYSICS_BLUR_DIR, num_samples=50)
    gaussian_test = get_test_pairs(GAUSSIAN_BLUR_DIR, num_samples=50)

    if not physics_test:
        print("❌ No physics-pipeline test pairs found. Run generate_blur_dataset.py first.")
        return
    if not gaussian_test:
        print("❌ No Gaussian-only test pairs found. Run train_vdsr_gaussian_only.py first.")
        return

    print(f"\n  Physics test set:  {len(physics_test)} pairs")
    print(f"  Gaussian test set: {len(gaussian_test)} pairs")

    # ── Models to evaluate ─────────────────────────────────────────────
    models = [
        ("Gaussian-Only VDSR", GAUSSIAN_ONLY_WEIGHT),
        ("Full-Physics VDSR",  FULL_PHYSICS_WEIGHT),
    ]

    # ── Results storage ────────────────────────────────────────────────
    # rows: (model_label, test_type, psnr, ssim, lpips, valid, total)
    all_results = []

    for label, weight_path in models:
        print(f"\n  {'─'*60}")
        print(f"  🔬 Evaluating: {label}")
        print(f"  {'─'*60}")

        # Test on physics pipeline
        print(f"  \n  [Test Set: Physics Pipeline]")
        r_phys = evaluate_model(weight_path, physics_test, label)
        all_results.append((label, "Physics Pipeline",
                            r_phys["psnr"], r_phys["ssim"], r_phys["lpips"],
                            r_phys["valid"], r_phys["total"]))

        # Test on Gaussian-only
        print(f"  \n  [Test Set: Gaussian-Only]")
        r_gauss = evaluate_model(weight_path, gaussian_test, label)
        all_results.append((label, "Gaussian-Only",
                            r_gauss["psnr"], r_gauss["ssim"], r_gauss["lpips"],
                            r_gauss["valid"], r_gauss["total"]))

        # Print inline summary
        print(f"\n  ┌──────────────────────┬───────────┬───────────┬───────────┐")
        print(f"  │ {'Test Set':<20} │ {'PSNR':>9} │ {'SSIM':>9} │ {'LPIPS':>9} │")
        print(f"  ├──────────────────────┼───────────┼───────────┼───────────┤")
        print(f"  │ {'Physics Pipeline':<20} │ {r_phys['psnr']:>9.2f} │ {r_phys['ssim']:>9.4f} │ {r_phys['lpips']:>9.4f} │")
        print(f"  │ {'Gaussian-Only':<20} │ {r_gauss['psnr']:>9.2f} │ {r_gauss['ssim']:>9.4f} │ {r_gauss['lpips']:>9.4f} │")
        print(f"  └──────────────────────┴───────────┴───────────┴───────────┘")

    # ── Master comparison table ────────────────────────────────────────
    print(f"\n\n  {'='*68}")
    print(f"  📊 MASTER COMPARISON — DOMAIN GAP")
    print(f"  {'='*68}")
    print(f"  {'Model':<25} {'Test Set':<20} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print(f"  {'-'*25} {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for label, test, p, s, l, v, t in all_results:
        lpips_str = f"{l:.4f}" if LPIPS_AVAILABLE else "N/A"
        weight_str = "Gaussian" if "Gaussian" in label else "Physics"
        print(f"  {weight_str:<25} {test:<20} {p:>8.2f} {s:>8.4f} {lpips_str:>8}")

    print(f"  {'-'*25} {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'='*68}")

    # ── Domain Gap Highlight ───────────────────────────────────────────
    # Find the key comparison: Gaussian-only model on physics test
    gauss_on_phys = [r for r in all_results if "Gaussian" in r[0] and "Physics" in r[1]]
    phys_on_phys  = [r for r in all_results if "Physics" in r[0] and "Physics" in r[1]]

    if gauss_on_phys and phys_on_phys:
        gp = gauss_on_phys[0]
        pp = phys_on_phys[0]
        psnr_drop = pp[2] - gp[2]
        ssim_drop = pp[3] - gp[3]
        print(f"\n  🔑 DOMAIN GAP QUANTIFIED:")
        print(f"     Gaussian-only VDSR on physics test:  PSNR {gp[2]:.2f}, SSIM {gp[3]:.4f}")
        print(f"     Full-physics VDSR on physics test:   PSNR {pp[2]:.2f}, SSIM {pp[3]:.4f}")
        print(f"     ΔPSNR: {psnr_drop:+.2f} dB  ΔSSIM: {ssim_drop:+.4f}")
        if psnr_drop < -2:
            print(f"     ✅ CONCLUSION: Domain gap confirmed — Gaussian-only model")
            print(f"        {'catastrophically fails' if psnr_drop < -5 else 'degrades significantly'}")
            print(f"        on physics-pipeline blurs.")
        else:
            print(f"     ⚠️  Domain gap smaller than expected. Consider more epochs or")
            print(f"        tighter degradation controls.")

    # ── Export CSV ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Test Set", "PSNR (dB)", "SSIM", "LPIPS",
                         "Valid Samples", "Total Samples"])
        for label, test, p, s, l, v, t in all_results:
            writer.writerow([
                label, test,
                f"{p:.2f}", f"{s:.4f}", f"{l:.4f}" if LPIPS_AVAILABLE else "N/A",
                v, t
            ])
    print(f"\n  💾 Results exported: {OUTPUT_CSV}")

    # ── Next steps ─────────────────────────────────────────────────────
    print(f"\n  📝 Next: Add this table to §4.2 (Experiment 1: The Domain Gap)")
    print(f"     in the paper to replace the qualitative claims with quantitative data.")
    print(f"     CSV is ready for import into your thesis spreadsheet.\n")


if __name__ == "__main__":
    main()
