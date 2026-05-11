# scripts/analyze_frequencies.py
#
# Quantifying the Hallucination Flaw via 2D Power Spectrum Analysis.
#
# Usage:
#   python scripts/analyze_frequencies.py \
#       --clear_dir   data/clear_test \
#       --vdsr_dir    outputs/vdsr \
#       --swinir_dir  outputs/swinir \
#       --esrgan_dir  outputs/esrgan \
#       --output      research/hallucination_power_spectrum.png
#
# This script computes the radially averaged 2D power spectrum (via FFT)
# for each set of images and plots comparative frequency profiles.
# The hypothesis: Real-ESRGAN will show elevated high-frequency energy
# relative to ground truth, while SwinIR will track the truth more closely.

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def radial_profile(power_spectrum):
    """Compute radially averaged power from a centered 2D power spectrum.

    Args:
        power_spectrum: 2D array (H, W) after fftshift — centre = DC component.

    Returns:
        radial_avg: 1D array of length min(H, W)//2.
        freq_bins:  1D array of normalised frequency values [0, 0.5].
    """
    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy)**2 + (x - cx)**2).astype(int)
    max_r = min(cy, cx)
    radial_avg = np.zeros(max_r)
    count      = np.zeros(max_r)
    for i in range(h):
        for j in range(w):
            ri = r[i, j]
            if ri < max_r:
                radial_avg[ri] += power_spectrum[i, j]
                count[ri]      += 1.0
    count = np.maximum(count, 1.0)
    radial_avg /= count
    freq_bins = np.linspace(0, 0.5, max_r)  # normalised frequency [0, Nyquist]
    return radial_avg, freq_bins


def compute_power_spectrum(image_path):
    """Load a single RGB image and return its average radial power spectrum.

    Args:
        image_path: Path to an image file (JPEG/PNG).

    Returns:
        radial_avg: 1D radial power spectrum.
        freq_bins: Corresponding frequency bins.
    """
    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")
    # Apply Hanning window to reduce spectral leakage
    h, w = img.shape
    hanning = np.outer(np.hanning(h), np.hanning(w))
    img_windowed = img.astype(np.float64) * hanning
    # 2D FFT
    fft = np.fft.fft2(img_windowed)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted)**2
    return radial_profile(power)


def load_images_from_dir(directory, max_images=None):
    """Load power spectra from all JPEG/PNG images in a directory.

    Returns:
        profiles: List of radial power spectra (one per image).
        freq:     Frequency bins (same for all).
    """
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif')
    paths = []
    for ext in extensions:
        paths.extend(glob(os.path.join(directory, ext)))
    if not paths:
        print(f"  !! No images found in {directory}")
        return [], None
    paths.sort()
    if max_images is not None:
        paths = paths[:max_images]
    print(f"  Loading {len(paths)} images from {directory} ...")
    profiles = []
    freq = None
    for p in paths:
        try:
            rad, freq = compute_power_spectrum(p)
            profiles.append(rad)
        except Exception as e:
            print(f"    Skipping {p}: {e}")
    return profiles, freq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FFT-based power spectrum analysis for hallucination detection."
    )
    parser.add_argument('--clear_dir',  type=str, required=True,
                        help='Directory of ground-truth clear images.')
    parser.add_argument('--vdsr_dir',   type=str, required=True,
                        help='Directory of VDSR-restored images.')
    parser.add_argument('--swinir_dir', type=str, required=True,
                        help='Directory of SwinIR-restored images.')
    parser.add_argument('--esrgan_dir', type=str, required=True,
                        help='Directory of Real-ESRGAN-restored images.')
    parser.add_argument('--output',     type=str, default='research/hallucination_power_spectrum.png',
                        help='Output path for the comparative plot.')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to load per directory.')
    args = parser.parse_args()

    print("=" * 60)
    print("Power Spectrum Analysis — Hallucination Quantification")
    print("=" * 60)

    # Load all power spectra
    clear_profiles, freq  = load_images_from_dir(args.clear_dir,  args.max_images)
    vdsr_profiles,   _    = load_images_from_dir(args.vdsr_dir,   args.max_images)
    swinir_profiles, _    = load_images_from_dir(args.swinir_dir, args.max_images)
    esrgan_profiles, _    = load_images_from_dir(args.esrgan_dir, args.max_images)

    if freq is None:
        print("!! No data loaded. Check directory paths.")
        sys.exit(1)

    # Convert to arrays and compute statistics
    def stats(profiles):
        if not profiles:
            return None, None, None
        arr = np.array(profiles)          # (N, max_r)
        mean = np.log10(np.mean(arr, axis=0) + 1e-12)
        std  = np.log10(np.std(arr,  axis=0) + 1e-12)
        return mean, std, arr.shape[0]

    clear_m,  clear_s,  n_clear  = stats(clear_profiles)
    vdsr_m,   vdsr_s,   n_vdsr   = stats(vdsr_profiles)
    swinir_m, swinir_s, n_swinir = stats(swinir_profiles)
    esrgan_m, esrgan_s, n_esrgan = stats(esrgan_profiles)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Limit frequency range to meaningful region (avoid Nyquist noise)
    max_freq_idx = len(freq)

    label_clear  = f"Ground Truth (n={n_clear})"  if n_clear  else "Ground Truth"
    label_vdsr   = f"VDSR (n={n_vdsr})"            if n_vdsr   else "VDSR"
    label_swinir = f"SwinIR (n={n_swinir})"        if n_swinir else "SwinIR"
    label_esrgan = f"Real-ESRGAN (n={n_esrgan})"   if n_esrgan else "Real-ESRGAN"

    if clear_m is not None:
        ax.plot(freq[:max_freq_idx], clear_m[:max_freq_idx],
                color='black', linewidth=2, label=label_clear)
        ax.fill_between(freq[:max_freq_idx],
                        clear_m[:max_freq_idx] - clear_s[:max_freq_idx],
                        clear_m[:max_freq_idx] + clear_s[:max_freq_idx],
                        color='black', alpha=0.1)

    if vdsr_m is not None:
        ax.plot(freq[:max_freq_idx], vdsr_m[:max_freq_idx],
                color='blue', linewidth=1.5, linestyle='--', label=label_vdsr)
        ax.fill_between(freq[:max_freq_idx],
                        vdsr_m[:max_freq_idx] - vdsr_s[:max_freq_idx],
                        vdsr_m[:max_freq_idx] + vdsr_s[:max_freq_idx],
                        color='blue', alpha=0.08)

    if swinir_m is not None:
        ax.plot(freq[:max_freq_idx], swinir_m[:max_freq_idx],
                color='green', linewidth=1.5, linestyle='-.', label=label_swinir)
        ax.fill_between(freq[:max_freq_idx],
                        swinir_m[:max_freq_idx] - swinir_s[:max_freq_idx],
                        swinir_m[:max_freq_idx] + swinir_s[:max_freq_idx],
                        color='green', alpha=0.08)

    if esrgan_m is not None:
        ax.plot(freq[:max_freq_idx], esrgan_m[:max_freq_idx],
                color='red', linewidth=1.5, linestyle=':', label=label_esrgan)
        ax.fill_between(freq[:max_freq_idx],
                        esrgan_m[:max_freq_idx] - esrgan_s[:max_freq_idx],
                        esrgan_m[:max_freq_idx] + esrgan_s[:max_freq_idx],
                        color='red', alpha=0.08)

    ax.set_xlabel("Normalised Spatial Frequency", fontsize=13)
    ax.set_ylabel("Log Power (a.u.)", fontsize=13)
    ax.set_title("2D Power Spectrum Comparison — Hallucination Analysis", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate the high-frequency region of interest
    ax.axvspan(0.3, 0.5, color='gray', alpha=0.06, label='High-freq region')
    ax.annotate('High-frequency region\n(Hallucination signature)',
                xy=(0.35, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] else 0),
                fontsize=10, fontstyle='italic', color='gray',
                ha='center')

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"\nOK Plot saved to: {args.output}")

    # Print summary statistics (high-frequency band)
    print("\n--- High-Frequency Energy Comparison (f > 0.3) ---")
    if clear_m is not None:
        hf_idx = np.where(freq >= 0.3)[0]
        def hf_energy(mean_arr):
            return np.mean(10**mean_arr[hf_idx]) if len(hf_idx) > 0 else 0.0
        e_clear  = hf_energy(clear_m)
        e_vdsr   = hf_energy(vdsr_m)   if vdsr_m   is not None else 0.0
        e_swinir = hf_energy(swinir_m) if swinir_m is not None else 0.0
        e_esrgan = hf_energy(esrgan_m) if esrgan_m is not None else 0.0
        print(f"  Ground Truth:           {e_clear:.2e}")
        print(f"  VDSR:                   {e_vdsr:.2e}   (ratio vs GT: {e_vdsr/e_clear:.2f}x)" if e_clear else "  VDSR: N/A")
        print(f"  SwinIR:                 {e_swinir:.2e}   (ratio vs GT: {e_swinir/e_clear:.2f}x)" if e_clear else "  SwinIR: N/A")
        print(f"  Real-ESRGAN:            {e_esrgan:.2e}   (ratio vs GT: {e_esrgan/e_clear:.2f}x)" if e_clear else "  Real-ESRGAN: N/A")
    print("\nDone.")


if __name__ == '__main__':
    main()
