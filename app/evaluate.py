# app/evaluate.py
#
# Standalone evaluation script for all restoration architectures.
# Computes PSNR, SSIM, and LPIPS against ground-truth clear images.
#
# Usage:
#   Evaluate a single model:
#       python app/evaluate.py --model SwinIR --weights swinir_hybrid.pth
#
#   Evaluate all available .pth files and export to CSV:
#       python app/evaluate.py --all --num_samples 50
#
#   Classical baselines (Wiener, RL):
#       python scripts/evaluate_baselines.py

import sqlite3
import cv2
import torch
import numpy as np
import os
import argparse
import csv
from datetime import datetime
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# LPIPS perceptual metric (optional)
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️ lpips not installed. Run: pip install lpips")

# Import all model architectures
from models import SimpleRestorationNet, SRCNN, VDSR, SwinIR, RRDBNet

# ── Known weight files and their model associations ──────────────────────
# Format: (filename_pattern, model_name, display_label)
# Order matters: first match wins.
KNOWN_WEIGHTS = [
    # SwinIR
    ("swinir",     "SwinIR",       "SwinIR"),
    # Real-ESRGAN
    ("esrgan",     "Real-ESRGAN",  "Real-ESRGAN"),
    # VDSR
    ("vdsr",       "VDSR",         "VDSR"),
    # SRCNN
    ("srcnn",      "SRCNN",        "SRCNN"),
    ("sCNN",       "Simple CNN",   "Simple CNN"),
    ("Simple",     "Simple CNN",   "Simple CNN"),
    # Fallback: try to guess from the model_metrics DB
]

RESULTS_DIR = "eval_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def detect_model_from_filename(filename):
    """Guess the model architecture from the .pth filename."""
    fname = filename.lower()
    for pattern, model_name, _ in KNOWN_WEIGHTS:
        if pattern.lower() in fname:
            return model_name
    return None


def create_model(model_name, device):
    """Instantiate the correct model architecture."""
    if model_name == "Simple CNN":
        return SimpleRestorationNet().to(device)
    elif model_name == "SRCNN":
        return SRCNN().to(device)
    elif model_name == "VDSR":
        return VDSR().to(device)
    elif model_name == "SwinIR":
        return SwinIR(img_size=128).to(device)
    elif model_name == "Real-ESRGAN":
        return RRDBNet().to(device)
    else:
        return None


def evaluate_model(model_name, weight_file, num_samples=50, csv_writer=None):
    """Run PSNR/SSIM/LPIPS evaluation on a trained model.

    Args:
        model_name: One of "Simple CNN", "SRCNN", "VDSR", "SwinIR", "Real-ESRGAN".
        weight_file: Path to the .pth weight file.
        num_samples: Number of test images to evaluate.
        csv_writer: Optional csv.writer to append results row.

    Returns:
        dict with keys: model, weights, psnr, ssim, lpips, samples, valid
    """
    result = {
        "model": model_name,
        "weights": os.path.basename(weight_file),
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
        "samples": num_samples,
        "valid": 0,
    }

    print(f"\n{'='*60}")
    print(f"📊 Evaluating: {model_name}")
    print(f"   Weights: {weight_file}")
    print(f"{'='*60}")

    # 1. Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    model = create_model(model_name, device)
    if model is None:
        print(f"   ❌ Unknown model: {model_name}")
        return result

    if not os.path.exists(weight_file):
        print(f"   ⚠️  {weight_file} not found! Skipping.")
        return result

    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.eval()
    print(f"   ✅ Model loaded successfully.")

    # Model-specific crop size (matching training_tab.py evaluation)
    # SwinIR uses img_size=128; Real-ESRGAN uses 96; others use 256
    if model_name == "SwinIR":
        crop_size = 128
    elif model_name == "Real-ESRGAN":
        crop_size = 96
    else:
        crop_size = 256


    # 2. Image Preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()
    ])

    # 3. Fetch Test Data from Database
    conn = sqlite3.connect('data/database.db')
    cursor = conn.cursor()
    # Use a deterministic offset based on the weight file hash to avoid random variation
    cursor.execute("SELECT COUNT(*) FROM samples")
    total = cursor.fetchone()[0]
    offset = max(0, total - num_samples - 10)
    cursor.execute(
        "SELECT blur_path, clear_path FROM samples ORDER BY id LIMIT ? OFFSET ?",
        (num_samples, offset)
    )
    test_pairs = cursor.fetchall()
    conn.close()
    print(f"   Test set: {len(test_pairs)} image pairs")

    # 4. Evaluation Loop
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    valid_samples = 0

    lpips_fn = lpips.LPIPS(net='alex').to(device) if LPIPS_AVAILABLE else None

    with torch.no_grad():
        for idx, (blur_path, clear_path) in enumerate(test_pairs):
            if not os.path.exists(blur_path) or not os.path.exists(clear_path):
                continue

            # Read images
            blur_img = cv2.imread(blur_path)
            clear_img = cv2.imread(clear_path)
            if blur_img is None or clear_img is None:
                continue

            # Convert to RGB
            blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
            clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)

            # Center-crop ground truth to match model input size (same as training_tab)
            clear_tensor = transform(clear_img)
            clear_img_resized = (clear_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # Run Inference
            input_tensor = transform(blur_img).unsqueeze(0).to(device)
            output_tensor = model(input_tensor)

            # Post-process tensor back to image array
            output_img = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            output_img = (np.clip(output_img, 0, 1) * 255).astype(np.uint8)

            # Calculate Metrics
            current_psnr = psnr(clear_img_resized, output_img, data_range=255)
            current_ssim = ssim(clear_img_resized, output_img, data_range=255,
                                channel_axis=-1, win_size=3)

            current_lpips = 0.0
            if LPIPS_AVAILABLE and lpips_fn is not None:
                clear_t = clear_tensor.unsqueeze(0).to(device)
                out_t = torch.from_numpy(output_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
                current_lpips = lpips_fn(out_t, clear_t).item()

            total_lpips += current_lpips
            total_psnr += current_psnr
            total_ssim += current_ssim
            valid_samples += 1

            if (idx + 1) % 25 == 0:
                print(f"   Processed {idx + 1}/{len(test_pairs)} images...")

    # 5. Results
    if valid_samples > 0:
        avg_psnr = total_psnr / valid_samples
        avg_ssim = total_ssim / valid_samples
        avg_lpips = total_lpips / valid_samples if LPIPS_AVAILABLE else 0.0

        result["psnr"] = avg_psnr
        result["ssim"] = avg_ssim
        result["lpips"] = avg_lpips
        result["valid"] = valid_samples

        print(f"\n   📈 RESULTS for {model_name}")
        print(f"   {'-'*40}")
        print(f"   Valid samples: {valid_samples}/{num_samples}")
        print(f"   PSNR: {avg_psnr:.2f} dB  (Higher is better)")
        print(f"   SSIM: {avg_ssim:.4f}      (Closer to 1.0 is better)")
        if LPIPS_AVAILABLE:
            print(f"   LPIPS: {avg_lpips:.4f}    (Lower is better)")
        print(f"   {'-'*40}")
    else:
        print("   ❌ No valid image pairs found.")

    # Write to CSV if writer provided
    if csv_writer is not None:
        csv_writer.writerow([
            model_name,
            os.path.basename(weight_file),
            f"{result['psnr']:.2f}",
            f"{result['ssim']:.4f}",
            f"{result['lpips']:.4f}" if LPIPS_AVAILABLE else "N/A",
            result["valid"],
            result["samples"],
        ])

    return result


def find_all_weights():
    """Search for .pth files in the project root and old weights/ directory."""
    pth_files = []

    # Search project root
    for f in os.listdir('.'):
        if f.endswith('.pth') and os.path.isfile(f):
            pth_files.append(f)

    # Search old weights/
    old_dir = 'old weights'
    if os.path.isdir(old_dir):
        for f in os.listdir(old_dir):
            if f.endswith('.pth'):
                pth_files.append(os.path.join(old_dir, f))

    return sorted(set(pth_files))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate restoration models with PSNR, SSIM, and LPIPS."
    )
    parser.add_argument('--model', type=str, default=None,
                        help='Model name: Simple CNN, SRCNN, VDSR, SwinIR, Real-ESRGAN')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to .pth weight file')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of test images (default: 50)')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all found .pth weight files')
    parser.add_argument('--output', type=str, default=None,
                        help='CSV output path (default: eval_results/eval_YYYY-MM-DD.csv)')
    args = parser.parse_args()

    # Determine output CSV path
    if args.output is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        args.output = os.path.join(RESULTS_DIR, f"eval_{date_str}.csv")

    # ── Mode 1: Evaluate all found .pth files ─────────────────────────
    if args.all:
        all_weights = find_all_weights()
        print(f"Found {len(all_weights)} .pth files to evaluate.")

        # Open CSV for writing
        with open(args.output, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Model", "Weights", "PSNR (dB)", "SSIM", "LPIPS",
                             "Valid Samples", "Total Samples"])

            results = []
            for w in all_weights:
                model_name = detect_model_from_filename(os.path.basename(w))
                if model_name is None:
                    print(f"   ⚠️  Could not detect model for {w}, skipping.")
                    continue
                result = evaluate_model(model_name, w, num_samples=args.num_samples,
                                        csv_writer=writer)
                results.append(result)

        print(f"\n✅ All results saved to: {args.output}")

        # Print summary table
        print(f"\n{'='*70}")
        print(f"{'📊 MASTER RESULTS TABLE':^70}")
        print(f"{'='*70}")
        print(f"{'Model':<20} {'Weights':<30} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
        print(f"{'-'*20} {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
        for r in results:
            if r["valid"] > 0:
                w_short = os.path.basename(r["weights"])[:28]
                lpips_str = f"{r['lpips']:.4f}" if LPIPS_AVAILABLE else "N/A"
                print(f"{r['model']:<20} {w_short:<30} {r['psnr']:>8.2f} {r['ssim']:>8.4f} {lpips_str:>8}")
        print(f"{'='*70}")

    # ── Mode 2: Evaluate a single model ───────────────────────────────
    elif args.model and args.weights:
        with open(args.output, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Model", "Weights", "PSNR (dB)", "SSIM", "LPIPS",
                             "Valid Samples", "Total Samples"])
            evaluate_model(args.model, args.weights, num_samples=args.num_samples,
                           csv_writer=writer)
        print(f"\n✅ Results saved to: {args.output}")

    else:
        print("Usage:")
        print("  Single model:  python app/evaluate.py --model SwinIR --weights swinir.pth")
        print("  Batch all:     python app/evaluate.py --all --num_samples 50")
        print("  Classical:     python scripts/evaluate_baselines.py")


if __name__ == "__main__":
    main()
