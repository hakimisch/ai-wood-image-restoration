#!/usr/bin/env python3
"""
scripts/batch_inference.py

Batch inference for all restoration architectures.
Generates restored images for FFT power spectrum analysis.
Saves outputs to subdirectories organized by model name.

Usage:
    python scripts/batch_inference.py --model SwinIR --weights swinir_hybrid.pth
    python scripts/batch_inference.py --model VDSR --weights vdsr_hybrid.pth
    python scripts/batch_inference.py --model Real-ESRGAN --weights esrgan_hybrid.pth

Default: processes the last 50 test images (matching evaluate.py).
"""

import os
import sys
import argparse
import sqlite3
import cv2
import torch
import numpy as np

# Add app/ to path for model imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app'))
from models import SimpleRestorationNet, SRCNN, VDSR, SwinIR, RRDBNet

DB_PATH = 'data/database.db'
OUTPUT_DIR = 'outputs'
NUM_IMAGES = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(model_name):
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


def process_in_patches(img_bgr, model, tile_size=128):
    """Alpha-blended tiled inference (same as generate_restored_dataset.py)."""
    h, w, c = img_bgr.shape
    overlap = tile_size // 4
    step = tile_size - overlap

    output_acc = np.zeros((h, w, c), dtype=np.float32)
    weight_acc = np.zeros((h, w, 1), dtype=np.float32)

    ramp = np.linspace(0, 1, overlap, endpoint=False, dtype=np.float32)
    ones = np.ones(tile_size - 2 * overlap, dtype=np.float32)
    ramp_1d = np.concatenate([ramp, ones, ramp[::-1]])[:tile_size]
    weight_2d = np.outer(ramp_1d, ramp_1d)[:, :, np.newaxis]

    y_starts = list(range(0, h - tile_size + 1, step))
    x_starts = list(range(0, w - tile_size + 1, step))
    if not y_starts or y_starts[-1] + tile_size < h:
        y_starts.append(max(0, h - tile_size))
    if not x_starts or x_starts[-1] + tile_size < w:
        x_starts.append(max(0, w - tile_size))

    for y0 in y_starts:
        y1 = min(y0 + tile_size, h)
        for x0 in x_starts:
            x1 = min(x0 + tile_size, w)
            tile = img_bgr[y0:y1, x0:x1]
            th, tw = tile.shape[:2]
            pad_h = tile_size - th
            pad_w = tile_size - tw
            if pad_h > 0 or pad_w > 0:
                tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

            # Convert BGR → RGB → tensor
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            input_tensor = torch.from_numpy(tile_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0

            with torch.no_grad():
                out_tensor = model(input_tensor)

            out_tile = out_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            out_tile = np.clip(out_tile, 0, 1)
            out_tile_bgr = cv2.cvtColor((out_tile * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)

            w2d = weight_2d.copy()
            if y0 == 0:    w2d[:overlap, :] = 1.0
            if y1 == h:    w2d[-overlap:, :] = 1.0
            if x0 == 0:    w2d[:, :overlap] = 1.0
            if x1 == w:    w2d[:, -overlap:] = 1.0
            w2d = w2d[:th, :tw]

            output_acc[y0:y1, x0:x1] += out_tile_bgr[:th, :tw] * w2d
            weight_acc[y0:y1, x0:x1] += w2d

    weight_acc = np.maximum(weight_acc, 1e-6)
    result = (output_acc / weight_acc).clip(0, 255).astype(np.uint8)

    # YCrCb Luminance Transfer
    orig_ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ai_ycc = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
    ai_ycc[:, :, 1] = orig_ycc[:, :, 1]
    ai_ycc[:, :, 2] = orig_ycc[:, :, 2]
    result = cv2.cvtColor(ai_ycc, cv2.COLOR_YCrCb2BGR)

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch inference for FFT hallucination analysis.")
    parser.add_argument('--model', type=str, required=True,
                        help='Model name: Simple CNN, SRCNN, VDSR, SwinIR, Real-ESRGAN')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to .pth weight file')
    parser.add_argument('--num_images', type=int, default=NUM_IMAGES,
                        help=f'Number of test images (default: {NUM_IMAGES})')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    args = parser.parse_args()

    # Resolve model name from filename if needed (support --model auto)
    model_name = args.model
    weight_file = args.weights

    print(f"{'='*60}")
    print(f"Batch Inference for: {model_name}")
    print(f"  Weights: {weight_file}")
    print(f"  Device:  {device}")
    print(f"{'='*60}")

    # 1. Load model
    model = create_model(model_name)
    if model is None:
        print(f"!! Unknown model: {model_name}")
        sys.exit(1)

    if not os.path.exists(weight_file):
        print(f"!! Weight file not found: {weight_file}")
        sys.exit(1)

    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.eval()
    print(f"OK Model loaded successfully.")

    # 2. Determine output subdirectory
    weights_stem = os.path.splitext(os.path.basename(weight_file))[0]
    model_tag = model_name.lower().replace(" ", "_").replace("-", "_")
    out_subdir = os.path.join(args.output_dir, f"{model_tag}_{weights_stem}")
    os.makedirs(out_subdir, exist_ok=True)

    # Also create clear_test directory if needed
    clear_dir = os.path.join(args.output_dir, "clear_gt")
    os.makedirs(clear_dir, exist_ok=True)

    # 3. Fetch test pairs from DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM samples")
    total = cursor.fetchone()[0]
    offset = max(0, total - args.num_images)
    cursor.execute(
        "SELECT blur_path, clear_path FROM samples ORDER BY id LIMIT ? OFFSET ?",
        (args.num_images, offset)
    )
    test_pairs = cursor.fetchall()
    conn.close()
    print(f"  Test set: {len(test_pairs)} image pairs")

    # 4. Run inference
    with torch.no_grad():
        for idx, (blur_path, clear_path) in enumerate(test_pairs):
            if not os.path.exists(blur_path) or not os.path.exists(clear_path):
                continue

            # Read blur image
            blur_bgr = cv2.imread(blur_path)
            if blur_bgr is None:
                continue

            # Run tiled inference
            restored_bgr = process_in_patches(blur_bgr, model)

            # Save restored image
            filename = os.path.basename(blur_path)
            out_path = os.path.join(out_subdir, filename)
            cv2.imwrite(out_path, restored_bgr)

            # Copy clear image to clear_gt (only once per unique filename)
            clear_filename = os.path.basename(clear_path)
            clear_out_path = os.path.join(clear_dir, clear_filename)
            if not os.path.exists(clear_out_path):
                clear_bgr = cv2.imread(clear_path)
                if clear_bgr is not None:
                    cv2.imwrite(clear_out_path, clear_bgr)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(test_pairs):
                print(f"  [{idx+1}/{len(test_pairs)}] Processed.")

    print(f"\nOK Done. Outputs saved to: {out_subdir}")
    print(f"   Clear references: {clear_dir}")


if __name__ == '__main__':
    main()
