#!/usr/bin/env python
"""Audit paper claims against eval_results CSV data."""

import csv

# Paper's Table 4.1 claims
paper_claims = {
    'Wiener Filter':      {'PSNR': 3.82,  'SSIM': 0.008, 'LPIPS': 0.879},
    'Richardson-Lucy':    {'PSNR': 3.97,  'SSIM': 0.007, 'LPIPS': 0.891},
    'SimpleRestorationNet': {'PSNR': 18.82, 'SSIM': 0.401, 'LPIPS': 0.489},
    'SRCNN':              {'PSNR': 19.37, 'SSIM': 0.428, 'LPIPS': 0.446},
    'VDSR':               {'PSNR': 18.75, 'SSIM': 0.446, 'LPIPS': 0.388},
    'SwinIR':             {'PSNR': 21.18, 'SSIM': 0.474, 'LPIPS': 0.272},
    'Real-ESRGAN':        {'PSNR': 21.22, 'SSIM': 0.484, 'LPIPS': 0.263},
}

# CSV best rows (selecting best 50-epoch hybrid or MSE for each model)
csv_models = {
    'SimpleRestorationNet': ('Simple CNN,50e_sCNN_mse_accum2_4_6.pth',         17.74, 0.3929, 0.4171),
    'SRCNN':              ('50e_srcnn_hybrid_accum2_4_10.pth',                 19.37, 0.4275, 0.3430),
    'VDSR':               ('50e_vdsr_hybrid_accum2_4_9.pth',                   18.75, 0.4463, 0.2851),
    'SwinIR':             ('final_swinir_hybrid_accum8_4_6.pth',               21.18, 0.4743, 0.2719),
    'Real-ESRGAN':        ('final_50e_esrgan_hybrid_accum8_4_14.pth',          21.22, 0.4839, 0.2626),
}

# Also check other VDSR entries for the LPIPS=0.388 claim
# VDSR CVS: LPIPS=0.2851 is the best hybrid. 0.388 doesn't appear anywhere
vdsr_check = [
    ('final_vdsr_hybrid.pth', 17.83, 0.4105, 0.3439),
    ('final_vdsr_mse_accum2_4_3.pth', 18.27, 0.4320, 0.3050),
    ('final_vdsr_mse_accum2_4_8.pth', 18.66, 0.4445, 0.2852),
    ('final_vdsr_hybrid_accum2_4_9.pth', 18.75, 0.4463, 0.2851),
]

print("=" * 80)
print("PAPER CLAIMS vs EVAL CSV — Systematic Audit")
print("=" * 80)

print(f"\n{'Model':<25} {'Metric':<8} {'Paper':<10} {'CSV':<10} {'Match':<8}")
print("-" * 65)

for model, claims in paper_claims.items():
    if model in csv_models:
        weight, csv_psnr, csv_ssim, csv_lpips = csv_models[model]
        for metric in ['PSNR', 'SSIM', 'LPIPS']:
            paper_val = claims[metric]
            csv_val = {'PSNR': csv_psnr, 'SSIM': csv_ssim, 'LPIPS': csv_lpips}[metric]
            
            # Determine tolerance
            if metric == 'PSNR':
                tol = 0.05
            elif metric == 'SSIM':
                tol = 0.002
            else:  # LPIPS
                tol = 0.005
            
            match = '✅' if abs(paper_val - csv_val) <= tol else '❌'
            diff = paper_val - csv_val
            
            print(f"{model:<25} {metric:<8} {paper_val:<10.3f} {csv_val:<10.4f} {match:<8} (Δ={diff:+.4f})")

print(f"\n\n=== Additional checks ===")

# Check all CSV LPIPS values for models with mismatches
print(f"\nAll VDSR LPIPS values in CSV:")
with open(r'C:\Hakimi\Internship\GUI Image Restoration\eval_results\eval_2026-05-06.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Model'] == 'VDSR':
            print(f"  {row['Weights']:<45} PSNR={row['PSNR (dB)']:<6} SSIM={row['SSIM']:<8} LPIPS={row['LPIPS']}")

print(f"\nAll SRCNN LPIPS values in CSV:")
with open(r'C:\Hakimi\Internship\GUI Image Restoration\eval_results\eval_2026-05-06.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Model'] == 'SRCNN':
            print(f"  {row['Weights']:<45} PSNR={row['PSNR (dB)']:<6} SSIM={row['SSIM']:<8} LPIPS={row['LPIPS']}")

print(f"\nAll Simple CNN LPIPS values in CSV:")
with open(r'C:\Hakimi\Internship\GUI Image Restoration\eval_results\eval_2026-05-06.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Model'] == 'Simple CNN':
            print(f"  {row['Weights']:<50} PSNR={row['PSNR (dB)']:<6} SSIM={row['SSIM']:<8} LPIPS={row['LPIPS']}")
