"""
scripts/batch_inference_for_fft.py

Generate correctly restored images for FFT analysis using FINAL 50-epoch weights.
Outputs to outputs/fft_analysis/ with matching filenames across all architectures.
"""
import os, sys, sqlite3, cv2, torch, numpy as np
from glob import glob

# Add app/ to path for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
from models import VDSR, SwinIR, RRDBNet

DB_PATH = 'data/database.db'
OUT_DIR = 'outputs/fft_analysis'
NUM_SAMPLES = 50
CROP_SIZES = {'VDSR': 256, 'SwinIR': 128, 'Real-ESRGAN': 96}
WEIGHTS = {
    'VDSR': '50e_vdsr_hybrid_accum2_4_9.pth',
    'SwinIR': 'swinir_hybrid_accum8_4_6.pth',
    'Real-ESRGAN': 'final_50e_esrgan_hybrid_accum8_4_14.pth',
}
MODEL_CLASSES = {
    'VDSR': lambda: VDSR(),
    'SwinIR': lambda: SwinIR(img_size=128),
    'Real-ESRGAN': lambda: RRDBNet(),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUT_DIR, exist_ok=True)

# --- Get test set ---
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM samples')
total = cursor.fetchone()[0]
offset = max(0, total - NUM_SAMPLES - 10)
cursor.execute('SELECT blur_path, clear_path, sample_name FROM samples ORDER BY id LIMIT ? OFFSET ?',
               (NUM_SAMPLES, offset))
test_pairs = cursor.fetchall()
conn.close()

print(f'Test set: {len(test_pairs)} image pairs')

# --- Helper: tiled inference (matching ai.py) ---
def tiled_inference(img_bgr, model, tile_size):
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
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            inp = torch.from_numpy(tile_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                out = model(inp)
            out_tile = out.squeeze().permute(1, 2, 0).cpu().numpy()
            out_tile = np.clip(out_tile, 0, 1)
            out_bgr = cv2.cvtColor((out_tile * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)
            w2d = weight_2d.copy()
            if y0 == 0: w2d[:overlap, :] = 1.0
            if y1 == h: w2d[-overlap:, :] = 1.0
            if x0 == 0: w2d[:, :overlap] = 1.0
            if x1 == w: w2d[:, -overlap:] = 1.0
            w2d = w2d[:th, :tw]
            output_acc[y0:y1, x0:x1] += out_bgr[:th, :tw] * w2d
            weight_acc[y0:y1, x0:x1] += w2d
    weight_acc = np.maximum(weight_acc, 1e-6)
    result = (output_acc / weight_acc).clip(0, 255).astype(np.uint8)
    # YCrCb luminance transfer
    orig_ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ai_ycc = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
    ai_ycc[:,:,1] = orig_ycc[:,:,1]
    ai_ycc[:,:,2] = orig_ycc[:,:,2]
    return cv2.cvtColor(ai_ycc, cv2.COLOR_YCrCb2BGR)

# --- Load models ---
models = {}
for name, wname in WEIGHTS.items():
    wpath = wname if os.path.exists(wname) else os.path.join('..', wname)
    if not os.path.exists(wpath):
        print(f'  !! {wname} not found')
        continue
    model = MODEL_CLASSES[name]().to(device)
    model.load_state_dict(torch.load(wpath, map_location=device, weights_only=True))
    model.eval()
    models[name] = model
    print(f'  Loaded {name} from {wname}')

# --- Create output subdirs ---
for name in ['clear', 'blurry'] + list(WEIGHTS.keys()):
    os.makedirs(os.path.join(OUT_DIR, name), exist_ok=True)

# --- Process each test image ---
for idx, (blur_path, clear_path, sample_name) in enumerate(test_pairs):
    if not os.path.exists(blur_path) or not os.path.exists(clear_path):
        continue
    blur_bgr = cv2.imread(blur_path)
    clear_bgr = cv2.imread(clear_path)
    if blur_bgr is None or clear_bgr is None:
        continue
    
    # Save blurry and clear (resized to 256x256 for consistency)
    h, w = clear_bgr.shape[:2]
    scale = min(256 / h, 256 / w) * 1.0
    new_h, new_w = int(h * scale), int(w * scale)
    blur_resized = cv2.resize(blur_bgr, (new_w, new_h)) if scale < 1.0 else blur_bgr
    clear_resized = cv2.resize(clear_bgr, (new_w, new_h)) if scale < 1.0 else clear_bgr
    
    out_name = f'{sample_name}.jpg'
    cv2.imwrite(os.path.join(OUT_DIR, 'blurry', out_name), blur_bgr)
    cv2.imwrite(os.path.join(OUT_DIR, 'clear', out_name), clear_bgr)
    
    for name in ['VDSR', 'SwinIR', 'Real-ESRGAN']:
        if name not in models:
            continue
        tile_size = CROP_SIZES[name]
        restored = tiled_inference(blur_bgr, models[name], tile_size)
        cv2.imwrite(os.path.join(OUT_DIR, name, out_name), restored)
    
    if (idx + 1) % 10 == 0:
        print(f'  [{idx+1}/{len(test_pairs)}] processed {sample_name}')

print(f'\nDone. Outputs in {OUT_DIR}/')
print('  Subdirs: clear, blurry, VDSR, SwinIR, Real-ESRGAN')
