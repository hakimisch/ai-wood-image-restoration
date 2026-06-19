# scripts/evaluate_restore_classify.py
#
# Restore → Classify Pipeline Evaluation
#
# Measures whether image restoration improves wood species classification
# accuracy by comparing classifier performance on blurred, restored,
# and clear versions of the same images.
#
# Usage:
#   Single (model, classifier) pair:
#     python scripts/evaluate_restore_classify.py \
#         --restore_model VDSR \
#         --restore_weights 50e_vdsr_hybrid_accum2_4_9.pth \
#         --classifier_weights 1-50e_resnet18_classifier_32batch_unfrozen_10_6.pth \
#         --classifier_backbone resnet18
#
#   Full sweep (all 5 restoration models × all 6 classifiers):
#     python scripts/evaluate_restore_classify.py --all
#
#   Generate restored images only (skip classification):
#     python scripts/evaluate_restore_classify.py --generate_only --restore_model VDSR \
#         --restore_weights 50e_vdsr_hybrid_accum2_4_9.pth
#
#   Classify existing restored images (no regeneration):
#     python scripts/evaluate_restore_classify.py --classify_only \
#         --restore_dir outputs/restore_classify/restored/VDSR \
#         --classifier_weights 1-50e_resnet18_classifier_32batch_unfrozen_10_6.pth \
#         --classifier_backbone resnet18

import os
import sys
import json
import csv
import argparse
import random
import sqlite3
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np

# Add project root and app/ to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'app'))

import torch
from models import SimpleRestorationNet, SRCNN, VDSR, SwinIR, RRDBNet
from classifier import create_classifier, detect_backbone_from_weights, VALID_BACKBONES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'database.db')
KAYU_DIR = os.path.join(PROJECT_ROOT, 'Kayu')
BLUR_DIR = os.path.join(PROJECT_ROOT, 'data', 'blurred')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'restore_classify')
RESTORED_DIR = os.path.join(OUTPUT_DIR, 'restored')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Restoration model registry
# ---------------------------------------------------------------------------

RESTORE_MODELS = {
    'SimpleCNN': {
        'builder': lambda: SimpleRestorationNet(),
        'crop_size': 256,
    },
    'SRCNN': {
        'builder': lambda: SRCNN(),
        'crop_size': 256,
    },
    'VDSR': {
        'builder': lambda: VDSR(),
        'crop_size': 256,
    },
    'SwinIR': {
        'builder': lambda: SwinIR(img_size=128),
        'crop_size': 128,
    },
    'Real-ESRGAN': {
        'builder': lambda: RRDBNet(),
        'crop_size': 96,
    },
}

# Known restoration weights (best 50-epoch files)
RESTORE_WEIGHT_MAP = {
    'SimpleCNN': '50e_sCNN_mse_accum2_4_6.pth',
    'SRCNN': '50e_srcnn_hybrid_accum2_4_10.pth',
    'VDSR': '50e_vdsr_hybrid_accum2_4_9.pth',
    'SwinIR': '50e_swinir_mse_accum8_4_7.pth',
    'Real-ESRGAN': '50e_esrgan_hybrid_accum8_4_14.pth',
}

# Classifier weights (all available 50-epoch runs)
CLASSIFIER_WEIGHT_MAP = [
    # (backbone, frozen, filename)
    ('resnet18', True,  '1-50e_resnet18_classifier_32batch_frozen_10_6.pth'),
    ('resnet18', False, '1-50e_resnet18_classifier_32batch_unfrozen_10_6.pth'),
    ('resnet50', True,  '50e_resnet50_classifier_32batch_frozen_10_6.pth'),
    ('resnet50', False, '50e_resnet50_classifier_32batch_unfrozen_10_6.pth'),
    ('swin_t',   False, '1-50e_swin_t_classifier_32batch_unfrozen_11_6.pth'),
]

# Score names for the three input conditions
INPUT_TYPES = ['blurred', 'restored', 'clear']


# ---------------------------------------------------------------------------
# Helper: Build classifier validation set with filenames
# ---------------------------------------------------------------------------

def build_validation_set(db_path=DB_PATH, kayu_dir=KAYU_DIR,
                         val_ratio=0.2, seed=42):
    """Build the exact same stratified validation split as the classifier
    training pipeline, but also keep the sample_name for file lookup.

    Returns:
        list of (sample_name, clear_path, label_index)
    """
    # 1. Build species index from DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT initials, full_name FROM species_registry ORDER BY initials")
    rows = cursor.fetchall()
    conn.close()

    seen = set()
    unique_species = []
    for initials, name in rows:
        if name not in seen:
            seen.add(name)
            unique_species.append((initials, name))

    name_to_idx = {name: i for i, (_, name) in enumerate(unique_species)}

    # 2. Collect all clear image paths
    all_samples = []  # (sample_name, clear_path, label)

    for species_name in os.listdir(kayu_dir):
        species_path = os.path.join(kayu_dir, species_name)
        if not os.path.isdir(species_path):
            continue
        if species_name not in name_to_idx:
            continue
        label = name_to_idx[species_name]

        for block in os.listdir(species_path):
            block_path = os.path.join(species_path, block)
            if not os.path.isdir(block_path):
                continue
            clear_dir = os.path.join(block_path, 'clear')
            if not os.path.isdir(clear_dir):
                continue

            for img_file in os.listdir(clear_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_path = os.path.join(clear_dir, img_file).replace('\\', '/')
                sample_name = os.path.splitext(img_file)[0]
                all_samples.append((sample_name, img_path, label))

    # 3. Stratified 80/20 split (mirroring classifier training)
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for sample_name, path, label in all_samples:
        by_label[label].append((sample_name, path, label))

    val_samples = []
    for label, samples in by_label.items():
        rng.shuffle(samples)
        n_val = max(1, int(len(samples) * val_ratio))
        val_samples.extend(samples[:n_val])

    print(f"  Validation set: {len(val_samples)} images, "
          f"{len(unique_species)} species")
    return val_samples, name_to_idx


# ---------------------------------------------------------------------------
# Helper: Preprocess image for restoration model
# ---------------------------------------------------------------------------

def preprocess_for_restoration(img_bgr, crop_size):
    """Convert BGR image → normalized CHW tensor, center-cropped to crop_size."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    top = max(0, (h - crop_size) // 2)
    left = max(0, (w - crop_size) // 2)
    cropped = img_rgb[top:top + crop_size, left:left + crop_size]
    tensor = torch.from_numpy(cropped.transpose(2, 0, 1)).float() / 255.0
    return tensor.unsqueeze(0)  # (1, C, H, W)


# ---------------------------------------------------------------------------
# Restoration inference
# ---------------------------------------------------------------------------

def restore_images(samples, restore_model_name, restore_weights_path,
                   output_subdir, crop_size):
    """Generate restored images for all validation samples.

    Args:
        samples: list of (sample_name, clear_path, label)
        restore_model_name: display name for logging
        restore_weights_path: path to .pth file
        output_subdir: directory to save restored images
        crop_size: model-specific input crop size

    Returns:
        dict mapping sample_name -> restored_image_path
    """
    os.makedirs(output_subdir, exist_ok=True)

    # Load restoration model
    print(f"\n  Loading restoration model: {restore_model_name}")
    print(f"    Weights: {restore_weights_path}")

    model_info = RESTORE_MODELS.get(restore_model_name)
    if model_info is None:
        print(f"    ❌ Unknown model: {restore_model_name}")
        return {}

    model = model_info['builder']().to(DEVICE)
    if not os.path.exists(restore_weights_path):
        print(f"    ❌ Weight file not found: {restore_weights_path}")
        return {}

    state_dict = torch.load(restore_weights_path, map_location=DEVICE,
                            weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"    ✅ Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Generate restored images
    result_map = {}
    total = len(samples)
    restored_count = 0

    with torch.no_grad():
        for i, (sample_name, clear_path, label) in enumerate(samples):
            # Derive blurred image path
            filename = os.path.basename(clear_path)
            blur_path = os.path.join(BLUR_DIR, filename)

            if not os.path.exists(blur_path):
                continue

            img_bgr = cv2.imread(blur_path)
            if img_bgr is None:
                continue

            # Run inference
            input_tensor = preprocess_for_restoration(img_bgr, crop_size).to(DEVICE)
            output_tensor = model(input_tensor)

            # Convert back to image
            output_img = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            output_img = (np.clip(output_img, 0, 1) * 255).astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

            # Save
            out_path = os.path.join(output_subdir, filename)
            cv2.imwrite(out_path, output_img)
            result_map[sample_name] = out_path
            restored_count += 1

            if (i + 1) % 200 == 0 or (i + 1) == total:
                print(f"    Restored [{i+1}/{total}] — {restored_count} written to disk")

    print(f"    ✅ Done: {restored_count}/{total} restored images saved to {output_subdir}")
    return result_map


# ---------------------------------------------------------------------------
# Classifier inference on a set of images
# ---------------------------------------------------------------------------

def classify_images(samples, classifier, image_path_fn, input_label):
    """Run classifier on a set of images.

    Args:
        samples: list of (sample_name, clear_path, label_index)
        classifier: SpeciesClassifier instance
        image_path_fn: callable(sample_name, clear_path) -> image_path or None
        input_label: 'blurred', 'restored', or 'clear' (for logging)

    Returns:
        dict: metrics, or None if all images missing
    """
    classifier.eval()
    classifier.to(DEVICE)

    correct = 0
    total = 0
    per_species = defaultdict(lambda: {'correct': 0, 'total': 0})
    all_confidences = []

    for i, (sample_name, clear_path, label) in enumerate(samples):
        img_path = image_path_fn(sample_name, clear_path)
        if img_path is None or not os.path.exists(img_path):
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # Predict
        try:
            pred_species, confidence, top3 = classifier.predict(img_bgr, top_k=3)
        except Exception as e:
            print(f"    ⚠️  Classifier error on {sample_name}: {e}")
            continue

        # Check correctness
        species_name = classifier.idx_to_name.get(label, 'Unknown')
        is_correct = (pred_species == species_name)
        if is_correct:
            correct += 1
        total += 1
        all_confidences.append(confidence)

        per_species[species_name]['correct'] += int(is_correct)
        per_species[species_name]['total'] += 1

    if total == 0:
        print(f"    ❌ No valid images for {input_label}")
        return None

    accuracy = correct / total
    mean_conf = np.mean(all_confidences) if all_confidences else 0.0

    # Per-species accuracy
    per_species_acc = {}
    for sp, counts in per_species.items():
        if counts['total'] > 0:
            per_species_acc[sp] = counts['correct'] / counts['total']

    result = {
        'input_type': input_label,
        'overall_acc': accuracy,
        'top3_acc': None,   # computed by caller if needed
        'mean_confidence': mean_conf,
        'correct': correct,
        'total': total,
        'per_species_acc': per_species_acc,
    }

    print(f"    {input_label:>10}: {accuracy:.2%} ({correct}/{total}) "
          f"conf={mean_conf:.3f}")
    return result


# ---------------------------------------------------------------------------
# Run a single (restoration, classifier) pair
# ---------------------------------------------------------------------------

def evaluate_pair(samples, restore_model_name, restore_weights_path,
                  classifier_weights_path, classifier_backbone,
                  reuse_restored=False):
    """Run the full restore→classify pipeline for one pair.

    Args:
        samples: validation set
        restore_model_name: None if no restoration (benchmark run)
        restore_weights_path: None if no restoration
        classifier_weights_path, classifier_backbone

    Returns:
        dict with results for blurred, restored, clear
    """
    results = {}

    # ── Step 1: Generate restored images (or reuse existing) ──────────
    restored_image_map = None
    if restore_model_name and restore_weights_path:
        model_dir_name = restore_model_name.lower()
        out_dir = os.path.join(RESTORED_DIR, model_dir_name)

        if reuse_restored and os.path.isdir(out_dir):
            # Reuse existing restored images
            print(f"  Reusing existing restored images from {out_dir}")
            restored_image_map = {}
            for sample_name, clear_path, label in samples:
                filename = os.path.basename(clear_path)
                rpath = os.path.join(out_dir, filename)
                if os.path.exists(rpath):
                    restored_image_map[sample_name] = rpath
            print(f"    Found {len(restored_image_map)} restored images")
        else:
            crop_size = RESTORE_MODELS[restore_model_name]['crop_size']
            restored_image_map = restore_images(
                samples, restore_model_name, restore_weights_path,
                out_dir, crop_size
            )
    else:
        print("  No restoration model — skipping restored condition")

    # ── Step 2: Load classifier ───────────────────────────────────────
    print(f"\n  Loading classifier: backbone={classifier_backbone}, "
          f"weights={classifier_weights_path}")
    classifier = create_classifier(
        weights_path=classifier_weights_path,
        backbone_name=classifier_backbone,
    )
    print(f"    ✅ Classifier loaded ({classifier.num_species} species)")

    # ── Step 3: Run on three input types ──────────────────────────────

    # 3a: Blurred
    def blurred_path_fn(sample_name, clear_path):
        return os.path.join(BLUR_DIR, os.path.basename(clear_path))
    print(f"\n  ── Running classification on blurred images ──")
    blurred_res = classify_images(samples, classifier, blurred_path_fn, 'blurred')
    if blurred_res:
        results['blurred'] = blurred_res

    # 3b: Restored
    if restored_image_map:
        def restored_path_fn(sample_name, clear_path):
            return restored_image_map.get(sample_name)
        print(f"  ── Running classification on restored images ──")
        restored_res = classify_images(samples, classifier, restored_path_fn, 'restored')
        if restored_res:
            results['restored'] = restored_res

    # 3c: Clear (ceiling)
    def clear_path_fn(sample_name, clear_path):
        return clear_path
    print(f"  ── Running classification on clear images (ceiling) ──")
    clear_res = classify_images(samples, classifier, clear_path_fn, 'clear')
    if clear_res:
        results['clear'] = clear_res

    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_results_csv(results_rows, output_path):
    """Write full results table."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        'restore_model', 'restore_weights',
        'classifier_backbone', 'classifier_weights_file', 'classifier_frozen',
        'input_type', 'overall_acc', 'mean_confidence',
        'correct', 'total', 'timestamp',
    ]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"\n✅ Results saved to: {output_path}")
    return output_path


def write_per_species_csv(results_by_input, classifier_backbone,
                          restore_model, output_dir):
    """Write per-species accuracy breakdown."""
    # Collect all species names
    all_species = set()
    for res in results_by_input.values():
        if res and res.get('per_species_acc'):
            all_species.update(res['per_species_acc'].keys())

    if not all_species:
        return

    path = os.path.join(
        output_dir,
        f"per_species_{classifier_backbone}_{restore_model or 'no_restore'}.csv"
    )
    fieldnames = ['species'] + INPUT_TYPES
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sp in sorted(all_species):
            row = {'species': sp}
            for it in INPUT_TYPES:
                res = results_by_input.get(it)
                if res and res.get('per_species_acc'):
                    row[it] = f"{res['per_species_acc'].get(sp, 0):.4f}"
                else:
                    row[it] = ''
            writer.writerow(row)
    print(f"  Per-species accuracy saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate restore→classify pipeline: measure classifier "
                    "accuracy on blurred vs restored vs clear images."
    )

    # Single-pair mode
    parser.add_argument('--restore_model', type=str, default=None,
                        choices=sorted(RESTORE_MODELS.keys()),
                        help='Restoration model name')
    parser.add_argument('--restore_weights', type=str, default=None,
                        help='Path to restoration .pth weight file')
    parser.add_argument('--classifier_weights', type=str, default=None,
                        help='Path to classifier .pth weight file')
    parser.add_argument('--classifier_backbone', type=str, default=None,
                        choices=sorted(VALID_BACKBONES) + [None],
                        help='Classifier backbone (auto-detect if None)')

    # Sweep mode
    parser.add_argument('--all', action='store_true',
                        help='Sweep all restoration models × all classifiers')

    # Partial modes
    parser.add_argument('--generate_only', action='store_true',
                        help='Only generate restored images, skip classification')
    parser.add_argument('--classify_only', action='store_true',
                        help='Skip restoration generation; use existing restored images')
    parser.add_argument('--restore_dir', type=str, default=None,
                        help='Directory of restored images for --classify_only')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path')

    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Build validation set ──────────────────────────────────────────
    print("=" * 65)
    print("Building classifier validation set (stratified 80/20 split)")
    print("=" * 65)
    samples, _ = build_validation_set()
    print(f"Total validation samples: {len(samples)}")

    # ── Mode: Generate only ───────────────────────────────────────────
    if args.generate_only:
        if not args.restore_model or not args.restore_weights:
            print("❌ --generate_only requires --restore_model and --restore_weights")
            return
        model_dir = os.path.join(RESTORED_DIR, args.restore_model.lower())
        crop_size = RESTORE_MODELS[args.restore_model]['crop_size']
        restore_images(samples, args.restore_model, args.restore_weights,
                       model_dir, crop_size)
        return

    # ── Mode: Classify only (reuse existing restored images) ──────────
    if args.classify_only:
        if not args.classifier_weights:
            print("❌ --classify_only requires --classifier_weights")
            return
        backbone = args.classifier_backbone
        if backbone is None:
            backbone = detect_backbone_from_weights(args.classifier_weights)
            print(f"  🔍 Auto-detected backbone: {backbone}")

        restore_dir = args.restore_dir
        if restore_dir is None:
            # Try to find from restore_model name
            restore_model_name = args.restore_model
            if restore_model_name:
                restore_dir = os.path.join(RESTORED_DIR, restore_model_name.lower())

        if restore_dir and os.path.isdir(restore_dir):
            print(f"\n📂 Using restored images from: {restore_dir}")
            restored_image_map = {}
            for sample_name, clear_path, label in samples:
                filename = os.path.basename(clear_path)
                rpath = os.path.join(restore_dir, filename)
                if os.path.exists(rpath):
                    restored_image_map[sample_name] = rpath
            print(f"   Found {len(restored_image_map)} restored images")
        else:
            print("⚠️  No restored images found. Run with --generate_only first.")
            restored_image_map = None

        # Load classifier and run
        print(f"\n🔬 Loading classifier: {args.classifier_weights}")
        classifier = create_classifier(
            weights_path=args.classifier_weights,
            backbone_name=backbone,
        )

        results_by_input = {}

        # Blurred
        def bpath(sn, cp): return os.path.join(BLUR_DIR, os.path.basename(cp))
        blurred_res = classify_images(samples, classifier, bpath, 'blurred')
        if blurred_res:
            results_by_input['blurred'] = blurred_res

        # Restored
        if restored_image_map:
            def rpath(sn, cp): return restored_image_map.get(sn)
            restored_res = classify_images(samples, classifier, rpath, 'restored')
            if restored_res:
                results_by_input['restored'] = restored_res

        # Clear
        def cpath(sn, cp): return cp
        clear_res = classify_images(samples, classifier, cpath, 'clear')
        if clear_res:
            results_by_input['clear'] = clear_res

        return

    # ── Mode: Full sweep ──────────────────────────────────────────────
    if args.all:
        run_plan = []
        for restore_name, default_weight in RESTORE_WEIGHT_MAP.items():
            weight_path = os.path.join(PROJECT_ROOT, default_weight)
            if os.path.exists(weight_path):
                for backbone, frozen, clf_file in CLASSIFIER_WEIGHT_MAP:
                    clf_path = os.path.join(PROJECT_ROOT, clf_file)
                    if os.path.exists(clf_path):
                        run_plan.append((restore_name, weight_path,
                                         clf_path, backbone, frozen))
                    else:
                        print(f"  ⚠️  Skipping missing classifier: {clf_file}")
            else:
                print(f"  ⚠️  Skipping missing restoration: {default_weight}")

        # Also add baseline runs (no restoration) for each classifier
        baseline_plan = []
        for backbone, frozen, clf_file in CLASSIFIER_WEIGHT_MAP:
            clf_path = os.path.join(PROJECT_ROOT, clf_file)
            if os.path.exists(clf_path):
                baseline_plan.append((None, None, clf_path, backbone, frozen))

        print(f"\n📋 Sweep plan: {len(run_plan)} restore×classifier pairs "
              f"+ {len(baseline_plan)} baseline runs")
        print(f"   Total: {len(run_plan) + len(baseline_plan)} evaluation runs")
        print(f"   Device: {DEVICE}\n")

        all_rows = []

        # First, generate all restored images (each model once)
        generated_models = set()
        for restore_name, weight_path, _, _, _ in run_plan:
            if restore_name not in generated_models:
                print(f"\n{'='*65}")
                print(f"GENERATING: {restore_name}")
                print(f"{'='*65}")
                model_dir = os.path.join(RESTORED_DIR, restore_name.lower())
                crop_size = RESTORE_MODELS[restore_name]['crop_size']
                restore_images(samples, restore_name, weight_path,
                               model_dir, crop_size)
                generated_models.add(restore_name)

        # Then, run classifier on each combination
        for restore_name, weight_path, clf_path, backbone, frozen in run_plan:
            print(f"\n{'='*65}")
            frozen_str = 'frozen' if frozen else 'unfrozen'
            restore_label = restore_name if restore_name else 'none'
            print(f"RUN: restore={restore_label} | "
                  f"classifier={backbone} ({frozen_str})")
            print(f"{'='*65}")

            # Build restored image map from disk
            model_dir = os.path.join(RESTORED_DIR, restore_name.lower())
            restored_map = {}
            if os.path.isdir(model_dir):
                for sn, cp, lb in samples:
                    fn = os.path.basename(cp)
                    rp = os.path.join(model_dir, fn)
                    if os.path.exists(rp):
                        restored_map[sn] = rp

            # Load classifier
            classifier = create_classifier(
                weights_path=clf_path,
                backbone_name=backbone,
            )

            # Run on three input types
            input_results = {}

            # Blurred
            def bp(sn, cp): return os.path.join(BLUR_DIR, os.path.basename(cp))
            input_results['blurred'] = classify_images(
                samples, classifier, bp, 'blurred')

            # Restored
            def rp(sn, cp): return restored_map.get(sn)
            input_results['restored'] = classify_images(
                samples, classifier, rp, 'restored')

            # Clear
            def cp(sn, cp_): return cp_
            input_results['clear'] = classify_images(
                samples, classifier, cp, 'clear')

            # Collect rows
            for input_type in INPUT_TYPES:
                res = input_results.get(input_type)
                if res is None:
                    continue
                all_rows.append({
                    'restore_model': restore_name or 'none',
                    'restore_weights': os.path.basename(weight_path) if weight_path else 'n/a',
                    'classifier_backbone': backbone,
                    'classifier_weights_file': os.path.basename(clf_path),
                    'classifier_frozen': int(frozen),
                    'input_type': input_type,
                    'overall_acc': f"{res['overall_acc']:.6f}",
                    'mean_confidence': f"{res['mean_confidence']:.6f}",
                    'correct': res['correct'],
                    'total': res['total'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })

            # Per-species CSV for this combo
            sp_dir = os.path.join(RESULTS_DIR, 'restore_classify')
            write_per_species_csv(input_results, backbone, restore_name, sp_dir)

        # ── Baseline runs (no restoration) ────────────────────────────
        for _, _, clf_path, backbone, frozen in baseline_plan:
            print(f"\n{'='*65}")
            frozen_str = 'frozen' if frozen else 'unfrozen'
            print(f"BASELINE: classifier={backbone} ({frozen_str})")
            print(f"{'='*65}")

            classifier = create_classifier(
                weights_path=clf_path,
                backbone_name=backbone,
            )

            input_results = {}

            def bp(sn, cp): return os.path.join(BLUR_DIR, os.path.basename(cp))
            input_results['blurred'] = classify_images(
                samples, classifier, bp, 'blurred')

            def cp(sn, cp_): return cp_
            input_results['clear'] = classify_images(
                samples, classifier, cp, 'clear')

            for input_type in ['blurred', 'clear']:
                res = input_results.get(input_type)
                if res is None:
                    continue
                all_rows.append({
                    'restore_model': 'none',
                    'restore_weights': 'n/a',
                    'classifier_backbone': backbone,
                    'classifier_weights_file': os.path.basename(clf_path),
                    'classifier_frozen': int(frozen),
                    'input_type': input_type,
                    'overall_acc': f"{res['overall_acc']:.6f}",
                    'mean_confidence': f"{res['mean_confidence']:.6f}",
                    'correct': res['correct'],
                    'total': res['total'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })

        # Write consolidated CSV
        output_path = args.output or os.path.join(
            RESULTS_DIR, 'restore_classify', 'restore_classify_full.csv')
        write_results_csv(all_rows, output_path)

        # Quick summary table
        print(f"\n{'='*65}")
        print(f"SUMMARY")
        print(f"{'='*65}")
        print(f"{'Restore':<14} {'Classifier':<18} {'Blurred':>10} {'Restored':>10} {'Clear':>10}")
        print(f"{'-'*14} {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
        for row in all_rows:
            if row['input_type'] == 'clear':
                # Print as summary line matching on restore_model + classifier
                pass  # We'll collect and print below

        # Better: group by (restore, classifier) and print one line
        grouped = {}
        for row in all_rows:
            key = (row['restore_model'], row['classifier_backbone'],
                   row['classifier_weights_file'])
            if key not in grouped:
                grouped[key] = {'blurred': '', 'restored': '', 'clear': ''}
            grouped[key][row['input_type']] = row['overall_acc']

        for key, vals in sorted(grouped.items()):
            restore_m, clf_back, clf_file = key
            short_clf = f"{clf_back}-{'F' if 'frozen' in clf_file else 'U'}"
            print(f"{restore_m:<14} {short_clf:<18} "
                  f"{vals.get('blurred', '-'):>10} "
                  f"{vals.get('restored', '-'):>10} "
                  f"{vals.get('clear', '-'):>10}")

        print(f"\n✅ Full sweep complete! Output: {output_path}")

    # ── Mode: Single pair ─────────────────────────────────────────────
    elif args.restore_model and args.restore_weights \
            and args.classifier_weights:
        backbone = args.classifier_backbone
        if backbone is None:
            backbone = detect_backbone_from_weights(args.classifier_weights)
            print(f"  🔍 Auto-detected backbone: {backbone}")

        results = evaluate_pair(
            samples, args.restore_model, args.restore_weights,
            args.classifier_weights, backbone,
            reuse_restored=False,
        )

        # Write output
        output_path = args.output or os.path.join(
            RESULTS_DIR, 'restore_classify',
            f"single_{args.restore_model.lower()}_{backbone}.csv")
        rows = []
        for input_type, res in results.items():
            if res:
                rows.append({
                    'restore_model': args.restore_model,
                    'restore_weights': os.path.basename(args.restore_weights),
                    'classifier_backbone': backbone,
                    'classifier_weights_file': os.path.basename(args.classifier_weights),
                    'classifier_frozen': 'unfrozen' if 'unfrozen' in args.classifier_weights else 'frozen',
                    'input_type': input_type,
                    'overall_acc': f"{res['overall_acc']:.6f}",
                    'mean_confidence': f"{res['mean_confidence']:.6f}",
                    'correct': res['correct'],
                    'total': res['total'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })
        write_results_csv(rows, output_path)

        # Per-species
        sp_dir = os.path.join(RESULTS_DIR, 'restore_classify')
        write_per_species_csv(results, backbone, args.restore_model, sp_dir)

        print(f"\n{'='*65}")
        print("SINGLE PAIR RESULTS")
        print(f"{'='*65}")
        for input_type in INPUT_TYPES:
            res = results.get(input_type)
            if res:
                print(f"  {input_type:>10}: {res['overall_acc']:.2%} "
                      f"({res['correct']}/{res['total']})")

    else:
        print("Usage:")
        print("  Full sweep:  python scripts/evaluate_restore_classify.py --all")
        print("  Single pair: python scripts/evaluate_restore_classify.py")
        print("                 --restore_model VDSR")
        print("                 --restore_weights 50e_vdsr_hybrid_accum2_4_9.pth")
        print("                 --classifier_weights <classifier.pth>")
        print("                 --classifier_backbone resnet18")
        print("  Generate:    python scripts/evaluate_restore_classify.py")
        print("                 --generate_only --restore_model VDSR")
        print("                 --restore_weights 50e_vdsr_hybrid_accum2_4_9.pth")
        print("  Classify:    python scripts/evaluate_restore_classify.py")
        print("                 --classify_only --restore_dir <path>")
        print("                 --classifier_weights <classifier.pth>")


if __name__ == '__main__':
    main()
