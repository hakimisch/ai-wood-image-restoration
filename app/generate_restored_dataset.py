import os
import sqlite3
import cv2
import torch
import numpy as np
import argparse 
from models import SwinIR 

DB_PATH = 'data/database.db'
RESTORED_OUT_DIR = 'data/restored'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESTORED_OUT_DIR, exist_ok=True)

# --- Helper Functions ---
def _to_tensor(img_bgr):
    """BGR uint8 numpy → CHW float32 [0,1] tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0

def process_in_patches(img_bgr, model, tile_size=128):
    """
    Replicates the Alpha-Blended Grid Stitching from ai.py 
    to prevent VRAM crashes and seam artifacts on full-res images.
    """
    h, w, c = img_bgr.shape
    overlap = tile_size // 4  # 32px overlap for 128px tiles
    step = tile_size - overlap

    output_acc = np.zeros((h, w, c), dtype=np.float32)
    weight_acc = np.zeros((h, w, 1), dtype=np.float32)

    ramp = np.linspace(0, 1, overlap, endpoint=False, dtype=np.float32)
    ones = np.ones(tile_size - 2 * overlap, dtype=np.float32)
    ramp_1d = np.concatenate([ramp, ones, ramp[::-1]])[:tile_size]
    weight_2d = np.outer(ramp_1d, ramp_1d)[:, :, np.newaxis]

    y_starts = list(range(0, h - tile_size + 1, step))
    x_starts = list(range(0, w - tile_size + 1, step))
    if not y_starts or y_starts[-1] + tile_size < h: y_starts.append(max(0, h - tile_size))
    if not x_starts or x_starts[-1] + tile_size < w: x_starts.append(max(0, w - tile_size))

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

            input_tensor = _to_tensor(tile).unsqueeze(0).to(device)
            with torch.no_grad():
                out_tensor = model(input_tensor)

            out_tile = out_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            out_tile = np.clip(out_tile, 0, 1)
            out_tile_bgr = cv2.cvtColor((out_tile * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)

            w2d = weight_2d.copy()
            if y0 == 0: w2d[:overlap, :] = 1.0
            if y1 == h: w2d[-overlap:, :] = 1.0
            if x0 == 0: w2d[:, :overlap] = 1.0
            if x1 == w: w2d[:, -overlap:] = 1.0

            w2d = w2d[:th, :tw]
            output_acc[y0:y1, x0:x1] += out_tile_bgr[:th, :tw] * w2d
            weight_acc[y0:y1, x0:x1] += w2d

    weight_acc = np.maximum(weight_acc, 1e-6)
    result = (output_acc / weight_acc).clip(0, 255).astype(np.uint8)
    
    # YCrCb Luminance Transfer (Ensures perfectly anchored colors/grayscale)
    orig_ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ai_ycc = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
    ai_ycc[:, :, 1] = orig_ycc[:, :, 1]
    ai_ycc[:, :, 2] = orig_ycc[:, :, 2]
    result = cv2.cvtColor(ai_ycc, cv2.COLOR_YCrCb2BGR)

    return result

def main():
    parser = argparse.ArgumentParser(description="Batch process blurry images through SwinIR.")
    parser.add_argument("--weights", "-w", type=str, required=True, help="Path to the .pth weights file.")
    parser.add_argument("--tile_size", "-t", type=int, default=128, help="Tile size for inference (default: 128)")
    args = parser.parse_args()

    # 1. Extract the name of the weights file for dynamic folder creation
    weights_name = os.path.splitext(os.path.basename(args.weights))[0]
    dynamic_out_dir = os.path.join(RESTORED_OUT_DIR, weights_name).replace('\\', '/')
    os.makedirs(dynamic_out_dir, exist_ok=True)

    print(f"🚀 Initializing SwinIR Master Engine (Tile Size: {args.tile_size})...")
    print(f"📁 Output will be saved to: {dynamic_out_dir}")

    model = SwinIR(img_size=args.tile_size).to(device)
    
    try:
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
        model.eval()
        print(f"✅ Weights loaded: {args.weights}")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return

    # --- 2. THE RESTORED DATABASE LOGIC ---
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(samples)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'restored_path' not in columns:
        cursor.execute("ALTER TABLE samples ADD COLUMN restored_path TEXT")
        conn.commit()
        print("Added 'restored_path' column to database.")

    # Only fetch rows that actually have a blur_path generated
    cursor.execute("SELECT id, blur_path FROM samples WHERE blur_path IS NOT NULL")
    rows = cursor.fetchall()
    total = len(rows)
    print(f"Processing {total} images through the AI...")
    # ----------------------------------------

    with torch.no_grad():
        for i, (sample_id, blur_path) in enumerate(rows):
            img_bgr = cv2.imread(blur_path)
            if img_bgr is None:
                continue

            # Run the heavy math
            restored_bgr = process_in_patches(img_bgr, model, tile_size=args.tile_size)

            # 3. Save the image into the dynamic subfolder
            filename = os.path.basename(blur_path)
            rest_path = os.path.join(dynamic_out_dir, filename).replace('\\', '/')
            cv2.imwrite(rest_path, restored_bgr)

            # 4. Update the database to point to this new specific path
            cursor.execute("UPDATE samples SET restored_path = ? WHERE id = ?", (rest_path, sample_id))

            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"  [{i+1}/{total}] Restored and saved.")
                conn.commit()

    conn.close()
    print(f"\n✨ Complete. Images saved to '{dynamic_out_dir}'.")

if __name__ == "__main__":
    main()