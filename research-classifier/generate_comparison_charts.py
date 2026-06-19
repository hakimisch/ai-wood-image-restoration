"""Generate comparison charts for the classifier research document."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# Figure 1: Frozen vs Unfrozen — Overall Accuracy (Val Set)
# ============================================================
backbones = ['ResNet18\n(11.2M params)', 'ResNet50\n(25.6M params)', 'Swin-T\n(28.3M params)']
frozen_acc  = [78.34, 86.48, 87.57]
unfrozen_acc = [99.93, 99.93, 99.85]

x = np.arange(len(backbones))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, frozen_acc, width, label='Frozen Backbone (ImageNet features)', color='#e74c3c', edgecolor='darkred', linewidth=0.5)
bars2 = ax.bar(x + width/2, unfrozen_acc, width, label='Unfrozen Backbone (Fine-tuned)', color='#27ae60', edgecolor='darkgreen', linewidth=0.5)

# Live camera annotations (from user testing)
live_annotations = {
    0: ('~40%', '~86%'),
    1: ('~45-55%', '~90%'),
    2: ('~60%', '~98%'),
}

for i, (f_bar, u_bar) in enumerate(zip(bars1, bars2)):
    ax.text(f_bar.get_x() + f_bar.get_width()/2, f_bar.get_height() + 1,
            f'{frozen_acc[i]:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#c0392b')
    ax.text(u_bar.get_x() + u_bar.get_width()/2, u_bar.get_height() + 1,
            f'{unfrozen_acc[i]:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1e8449')
    # Live camera annotation below bars
    live_f, live_u = live_annotations[i]
    ax.annotate(f'Live: {live_f}', xy=(f_bar.get_x() + f_bar.get_width()/2, 5),
                ha='center', va='bottom', fontsize=7.5, color='#c0392b', fontstyle='italic')
    ax.annotate(f'Live: {live_u}', xy=(u_bar.get_x() + u_bar.get_width()/2, 3),
                ha='center', va='bottom', fontsize=7.5, color='#1e8449', fontstyle='italic')

# Horizontal line at 100%
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4)

ax.set_ylabel('Validation Accuracy (%)', fontsize=11)
ax.set_title('Frozen vs Unfrozen: Three Backbones — Kayu/ Validation vs Live Camera', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(backbones, fontsize=10)
ax.set_ylim(0, 110)
ax.legend(fontsize=9, loc='lower right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('research-classifier/frozen_vs_unfrozen_comparison.png', dpi=200)
plt.close()
print("✅ Saved: research-classifier/frozen_vs_unfrozen_comparison.png")


# ============================================================
# Figure 2: Delta Gap — The Domain Gap Magnitude
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4))

delta_val = [99.93 - 78.34, 99.93 - 86.48, 99.85 - 87.57]
delta_live = [86 - 40, 90 - 50, 98 - 60]  # approximate from user's data

x = np.arange(len(backbones))
width = 0.35

bars_val = ax.bar(x - width/2, delta_val, width, label='Val Set Gap (Kayu/)', color='#3498db', edgecolor='#2471a3', linewidth=0.5)
bars_live = ax.bar(x + width/2, delta_live, width, label='Live Camera Gap', color='#9b59b6', edgecolor='#7d3c98', linewidth=0.5)

for bar, val in zip(bars_val, delta_val):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'+{val:.1f}pp', ha='center', fontsize=9, fontweight='bold', color='#2471a3')
for bar, val in zip(bars_live, delta_live):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'+{val}pp', ha='center', fontsize=9, fontweight='bold', color='#7d3c98')

ax.set_ylabel('Accuracy Improvement (percentage points)', fontsize=11)
ax.set_title('The Frozen→Unfrozen Lift: Validation vs Live Camera', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(backbones, fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('research-classifier/frozen_unfrozen_delta_gap.png', dpi=200)
plt.close()
print("✅ Saved: research-classifier/frozen_unfrozen_delta_gap.png")


# ============================================================
# Figure 3: Frozen Per-Species Accuracy — Hardest Species
# ============================================================
species_names = [
    'Balau', 'Bintangor', 'Chengal', 'Durian', 'Giam', 'Gerutu',
    'Jelutong', 'Kasai', 'Kedondong', 'Keranji', 'Kelat',
    'Meranti Bakau', 'Machang', 'Medang', 'Melunak', 'Meranti Dark Red',
    'Merawan', 'Merbau', 'Mersawa', 'Melantai', 'Mata Ulat',
    'Meranti White', 'Meranti Yellow', 'Pulai', 'Punah', 'Perupok',
    'Ramin', 'Rengas', 'Resak', 'Simpoh', 'Sepetir',
    'Tembusu', 'Terentang', 'Tualang'
]

# Per-species F1 from frozen models
f1_resnet18_frozen = [
    0.6579, 0.9176, 0.8333, 0.8158, 0.6923, 0.7895,
    0.8293, 0.6957, 0.6286, 0.8675, 0.7711, 0.7021,
    0.7765, 0.6032, 0.5479, 0.6667, 0.6374, 0.8831,
    0.8889, 0.6364, 0.9500, 0.7778, 0.8193, 0.9512,
    0.7778, 0.8537, 0.9412, 0.5600, 0.6667, 0.9750,
    0.7619, 1.0000, 0.7901, 0.8706
]

f1_resnet50_frozen = [
    0.7838, 0.9412, 0.9114, 0.8861, 0.8864, 0.8608,
    0.9383, 0.8861, 0.7619, 0.9412, 0.8750, 0.7467,
    0.8471, 0.6933, 0.7368, 0.7955, 0.6585, 0.8800,
    0.8916, 0.7778, 0.9877, 0.8941, 0.8642, 0.9639,
    0.9620, 0.9136, 0.9756, 0.7294, 0.8537, 0.9524,
    0.8537, 1.0000, 0.9231, 0.8000
]

f1_swin_t_frozen = [
    0.7895, 0.9512, 0.8736, 0.8800, 0.8000, 0.8293,
    0.9610, 0.8378, 0.8736, 0.9535, 0.9211, 0.7229,
    0.9302, 0.8276, 0.7632, 0.8571, 0.7200, 0.9639,
    0.9744, 0.6849, 0.9877, 0.8571, 0.9114, 0.9877,
    0.8916, 0.9383, 1.0000, 0.7778, 0.8000, 0.9512,
    0.7816, 1.0000, 0.8395, 0.8989
]

# Sort by average F1 across all 3 frozen models
avg_f1 = [(a + b + c) / 3 for a, b, c in zip(f1_resnet18_frozen, f1_resnet50_frozen, f1_swin_t_frozen)]
sorted_idx = np.argsort(avg_f1)
sorted_species = [species_names[i] for i in sorted_idx]
sorted_r18 = [f1_resnet18_frozen[i] for i in sorted_idx]
sorted_r50 = [f1_resnet50_frozen[i] for i in sorted_idx]
sorted_swin = [f1_swin_t_frozen[i] for i in sorted_idx]

fig, ax = plt.subplots(figsize=(10, 7))
y = np.arange(len(sorted_species))
height = 0.25

ax.barh(y - height, sorted_r18, height, label='ResNet18 Frozen', color='#e74c3c', alpha=0.8)
ax.barh(y, sorted_r50, height, label='ResNet50 Frozen', color='#3498db', alpha=0.8)
ax.barh(y + height, sorted_swin, height, label='Swin-T Frozen', color='#2ecc71', alpha=0.8)

ax.set_yticks(y)
ax.set_yticklabels(sorted_species, fontsize=8)
ax.set_xlabel('F1-Score', fontsize=11)
ax.set_title('Per-Species F1: Frozen Backbones (Sorted by Avg F1)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim(0, 1.05)
ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.4, label='80% Threshold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('research-classifier/per_species_frozen_f1_comparison.png', dpi=200)
plt.close()
print("✅ Saved: research-classifier/per_species_frozen_f1_comparison.png")


# ============================================================
# Figure 4: Unfrozen Per-Species — Ceiling Effect
# ============================================================
# All unfrozen models get ~100% for almost every species
# Show the tiny differences
fig, ax = plt.subplots(figsize=(10, 5))

# Find species where any unfrozen model wasn't perfect
unfrozen_imperfect = {
    'Melunak': (0.9873, 0.9873, 0.9873),
    'Kasai': (1.0, 1.0, 0.9877),
    'Terentang': (0.9877, 0.9877, 0.9877),
    'Tualang': (0.9885, 1.0, 0.9882),
}
imperfect_species = list(unfrozen_imperfect.keys())
r18_u = [unfrozen_imperfect[s][0] for s in imperfect_species]
r50_u = [unfrozen_imperfect[s][1] for s in imperfect_species]
swin_u = [unfrozen_imperfect[s][2] for s in imperfect_species]

x = np.arange(len(imperfect_species))
width = 0.25

ax.bar(x - width, [v * 100 for v in r18_u], width, label='ResNet18 Unfrozen', color='#e74c3c', alpha=0.8)
ax.bar(x, [v * 100 for v in r50_u], width, label='ResNet50 Unfrozen', color='#3498db', alpha=0.8)
ax.bar(x + width, [v * 100 for v in swin_u], width, label='Swin-T Unfrozen', color='#2ecc71', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(imperfect_species, fontsize=11)
ax.set_ylabel('F1-Score (%)', fontsize=11)
ax.set_title('Unfrozen Models: The Only Species With Imperfect Scores (34/35 at 100%)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(95, 101)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('research-classifier/unfrozen_ceiling_imperfect.png', dpi=200)
plt.close()
print("✅ Saved: research-classifier/unfrozen_ceiling_imperfect.png")


# ============================================================
# Figure 5: The Architectural Progression (Frozen Only)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

arches = ['ResNet18\n(11.2M)', 'ResNet50\n(25.6M)', 'Swin-Tiny\n(28.3M)']
frozen_vals = [78.34, 86.48, 87.57]

colors = ['#e74c3c', '#e67e22', '#2ecc71']
bars = ax.bar(arches, frozen_vals, color=colors, edgecolor='gray', linewidth=0.5, width=0.5)

for bar, val in zip(bars, frozen_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel('Validation Accuracy (%)', fontsize=11)
ax.set_title('Frozen Backbone Progression: ImageNet Features on Wood Microscopy', fontsize=12, fontweight='bold')
ax.set_ylim(0, 100)
ax.axhline(y=78.34, color='#e74c3c', linestyle='--', alpha=0.3, xmin=0, xmax=0.33)
ax.axhline(y=86.48, color='#e67e22', linestyle='--', alpha=0.3, xmin=0.33, xmax=0.66)
ax.axhline(y=87.57, color='#2ecc71', linestyle='--', alpha=0.3, xmin=0.66, xmax=1.0)
ax.grid(axis='y', alpha=0.3)

# Annotation
ax.annotate('Deeper → More transferable features\nbut all plateau < 90%',
            xy=(1, 92), fontsize=9, ha='center', color='#555',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

plt.tight_layout()
plt.savefig('research-classifier/frozen_architectural_progression.png', dpi=200)
plt.close()
print("✅ Saved: research-classifier/frozen_architectural_progression.png")


# ============================================================
# Copy confusion matrices to research-classifier/
# ============================================================
import shutil, os, glob

src_reports = 'reports'
dst = 'research-classifier'

for f in glob.glob(f'{src_reports}/confusion_matrix_*.png'):
    shutil.copy2(f, dst)
    print(f"✅ Copied: {f} → {dst}")

for f in glob.glob(f'{src_reports}/per_species_accuracy_*.png'):
    shutil.copy2(f, dst)
    print(f"✅ Copied: {f} → {dst}")

print("\n🎉 All classifier comparison charts generated!")
