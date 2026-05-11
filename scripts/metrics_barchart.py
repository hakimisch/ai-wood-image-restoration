"""
scripts/metrics_barchart.py

Generate metric comparison bar chart from evaluate.py CSV output.
"""
import os, sys, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Read the latest CSV
csv_dir = 'eval_results'
csv_files = sorted([f for f in os.listdir(csv_dir) if f.startswith('eval_') and f.endswith('.csv')])
if not csv_files:
    print('No CSV files found')
    sys.exit(1)

latest_csv = os.path.join(csv_dir, csv_files[-1])
print(f'Reading: {latest_csv}')

# Parse CSV - find best 50-epoch hybrid weights for each architecture
arch_configs = {
    'Simple CNN': {'weight': '50e_sCNN_mse_accum2_4_6.pth'},
    'SRCNN': {'weight': '50e_srcnn_hybrid_accum2_4_10.pth'},
    'VDSR': {'weight': '50e_vdsr_hybrid_accum2_4_9.pth'},
    'SwinIR': {'weight': 'final_swinir_hybrid_accum8_4_6.pth'},
    'Real-ESRGAN': {'weight': 'final_50e_esrgan_hybrid_accum8_4_14.pth'},
}

with open(latest_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for arch, cfg in arch_configs.items():
            if row['Weights'] == cfg['weight']:
                cfg['psnr'] = float(row['PSNR (dB)'])
                cfg['ssim'] = float(row['SSIM'])
                cfg['lpips'] = float(row['LPIPS'])

# Also read classical baselines
baseline_csv = 'scripts/baseline_results.csv'
baselines = {}
if os.path.exists(baseline_csv):
    with open(baseline_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            baselines[row['method']] = {
                'psnr': float(row['psnr']),
                'ssim': float(row['ssim']),
                'lpips': float(row['lpips'])
            }

# --- PSNR Bar Chart ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

arch_names = list(arch_configs.keys())
psnr_vals = [arch_configs[a].get('psnr', 0) for a in arch_names]
ssim_vals = [arch_configs[a].get('ssim', 0) for a in arch_names]
lpips_vals = [arch_configs[a].get('lpips', 0) for a in arch_names]

colors = ['#8ecfc9', '#ffbe7a', '#fa7f6f', '#82b0d2', '#beb8dc']

ax = axes[0]
bars = ax.bar(arch_names, psnr_vals, color=colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, psnr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.2f}',
            ha='center', va='bottom', fontsize=9)
ax.set_ylabel('PSNR (dB)', fontsize=11)
ax.set_title('Peak Signal-to-Noise Ratio', fontsize=12, fontweight='bold')
ax.set_xticklabels(arch_names, rotation=25, ha='right', fontsize=9)
ax.set_ylim(0, max(psnr_vals) * 1.2)
# Add Wiener/RL baseline
ax.axhline(y=baselines.get('Wiener', {}).get('psnr', 3.56), color='gray', linestyle='--', linewidth=1, alpha=0.6)
wiener_psnr = baselines.get('Wiener', {}).get('psnr', 3.56)
ax.text(0, wiener_psnr + 0.3, f'Wiener/RL: {wiener_psnr:.2f} dB',
        fontsize=8, color='gray')

ax = axes[1]
bars = ax.bar(arch_names, ssim_vals, color=colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, ssim_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.4f}',
            ha='center', va='bottom', fontsize=9)
ax.set_ylabel('SSIM', fontsize=11)
ax.set_title('Structural Similarity Index', fontsize=12, fontweight='bold')
ax.set_xticklabels(arch_names, rotation=25, ha='right', fontsize=9)
ax.set_ylim(0, max(ssim_vals) * 1.25)
ax.axhline(y=baselines.get('Wiener', {}).get('ssim', 0.0065), color='gray', linestyle='--', linewidth=1, alpha=0.6)

ax = axes[2]
bars = ax.bar(arch_names, lpips_vals, color=colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, lpips_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.4f}',
            ha='center', va='bottom', fontsize=9)
ax.set_ylabel('LPIPS', fontsize=11)
ax.set_title('Learned Perceptual Image Patch Similarity', fontsize=12, fontweight='bold')
ax.set_xticklabels(arch_names, rotation=25, ha='right', fontsize=9)
ax.set_ylim(0, max(lpips_vals) * 1.25)
ax.axhline(y=baselines.get('Wiener', {}).get('lpips', 0.8322), color='gray', linestyle='--', linewidth=1, alpha=0.6)

plt.tight_layout()
os.makedirs('research', exist_ok=True)
out_path = 'research/metrics_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.close()

print(f'\nTable 4.1 reference values:')
header = f'{"Model":<20} {"PSNR":>8} {"SSIM":>8} {"LPIPS":>8}'
print(header)
sep = "-" * 44
print(sep)
for a, p, s, l in zip(arch_names, psnr_vals, ssim_vals, lpips_vals):
    print(f'{a:<20} {p:>8.2f} {s:>8.4f} {l:>8.4f}')
print(sep)
w_psnr = baselines.get("Wiener", {}).get("psnr", 0)
w_ssim = baselines.get("Wiener", {}).get("ssim", 0)
w_lpips = baselines.get("Wiener", {}).get("lpips", 0)
rl_psnr = baselines.get("Richardson-Lucy", {}).get("psnr", 0)
rl_ssim = baselines.get("Richardson-Lucy", {}).get("ssim", 0)
rl_lpips = baselines.get("Richardson-Lucy", {}).get("lpips", 0)
print(f'Wiener Filter    {w_psnr:>8.2f} {w_ssim:>8.4f} {w_lpips:>8.4f}')
print(f'Richardson-Lucy  {rl_psnr:>8.2f} {rl_ssim:>8.4f} {rl_lpips:>8.4f}')
