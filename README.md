# 🌳 CAIRO Wood Image Restoration & Recognition Pipeline

> **An end-to-end AI pipeline for wood microscopic image restoration, species classification, and domain-gap research** — from automated data acquisition through physics-based degradation modeling, Vision Transformer restoration, and 3-backbone classifier evaluation. Developed at CAIRO Lab, Universiti Teknologi Malaysia.

---

## 📊 Project Overview

| Attribute | Detail |
|-----------|--------|
| **Goal** | Restoration of blur-degraded wood microscope images + species classification across 35 Malaysian hardwoods |
| **Status** | **Pipeline complete** — acquisition ✅ → physics degradation ✅ → 5-model restoration ✅ → 3-backbone classifier ✅ → evaluation ✅ → first-author paper 📄 |
| **Dataset** | **6,842+** paired clear/blur images across **35** wood species |
| **Hardware** | Intel i7-10750H + **6GB GTX 1660 Ti** (consumer laptop GPU) |
| **Best PSNR** | **21.22 dB** (Real-ESRGAN), **21.20 dB** (SwinIR) — vs 3.56 dB Wiener baseline |
| **Classifier** | **99.85% accuracy** (F1 ≥ 0.975) across 35 species |
| **Paper** | First-author thesis documenting domain-gap strategies, **14,600 words**, **4.2 GB** reproducible artifacts |

---

## 📐 Pipeline Architecture

```
                    ┌─────────────────────────────┐
                    │  DATA ACQUISITION (PyQt6)    │
                    │  USB Microscope → 30+ FPS    │
                    │  6,842 paired images, 35 spp │
                    └──────────┬──────────────────┘
                               │ Clear/Blur pairs
                               ▼
            ┌──────────────────────────────────────┐
            │  PHYSICS-BASED DEGRADATION PIPELINE   │
            │  Defocus · Motion Blur · Sensor Noise │
            │  LED banding · ISP · JPEG · Gamma     │
            │  Bridging synthetic-to-real gap       │
            └──────────────────┬───────────────────┘
                               │ Degraded images
                               ▼
       ┌────────────────────────────────────────────────┐
       │         RESTORATION (5 Architectures)           │
       │  ┌─────────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌───┐ │
       │  │ Simple  │ │SRCNN │ │VDSR │ │SwinIR│ │ESR│ │
       │  │ CNN     │ │      │ │     │ │(ViT) │ │GAN│ │
       │  │ 1.2K    │ │ 59K  │ │667K │ │4.1M  │ │17M│ │
       │  │ params  │ │      │ │     │ │      │ │   │ │
       │  └─────────┘ └──────┘ └──────┘ └──────┘ └───┘ │
       │  Selected: SwinIR (PSNR 21.20 dB)              │
       └──────────────────────┬────────────────────────┘
                              │ Restored images
                              ▼
       ┌────────────────────────────────────────────────┐
       │          CLASSIFIER (3 Backbones)               │
       │  ResNet18 (11.2M) · ResNet50 (25.6M)           │
       │  Swin-Tiny (28.3M)                             │
       │  99.85% accuracy · F1 ≥ 0.975                   │
       │  Swin-T ~98% on live camera → production choice │
       └────────────────────────────────────────────────┘
```

---

## 📊 Dataset

Acquired via a custom PyQt6 multi-threaded GUI that eliminates live-microscope lag:

| Metric | Value |
|--------|-------|
| **Paired images (clear/blur)** | **6,842** (6,915 DB rows) |
| **Wood species** | **35** Malaysian hardwoods |
| **Images per species** | ~200 (range 150–250) |
| **Training samples** | ~5,500 (80% split) |
| **Validation samples** | **1,376** (20% split) |
| **Evaluation holdout** | **50** images (deterministic seed 42) |
| **Acquisition rate** | **≥30 FPS** (multi-threaded PyQt6/OpenCV) |
| **Live sharpness metric** | Variance of Laplacian (VOL) — refresh every **50ms** |
| **Storage** | Semi-flat: `Kayu/Species/BlockID/clear/` — PyTorch `ImageFolder` compatible |

### Species Coverage

35 species fully mapped with scientific names (e.g., *Koompassia excelsa* for Tualang, *Intsia palembanica* for Merbau, *Hopea odorata* for Chengal) in SQLite registry.

---

## 🔬 Restoration — 5 Architectures Benchmarked

All models trained from scratch on the **6GB GTX 1660 Ti** — overcoming a **3–4× VRAM shortfall** via gradient accumulation, dynamic resolution scaling, and a 2GB in-memory RAM cache.

### Model Comparison (50-image holdout, standardized eval)

| Model | Parameters | Input Size | PSNR (dB) | SSIM | LPIPS | VRAM Strategy |
|-------|-----------|------------|:---------:|:----:|:-----:|---------------|
| **Real-ESRGAN** 🏆 | **16.7M** | 96×96 | **21.22** | **0.484** | **0.263** | Grad accum 8, AMP |
| **SwinIR** 🏆 | **4.1M** | 128×128 | **21.20** | 0.473 | 0.272 | Grad accum 8, dynamic crops |
| **SRCNN** | 59K | 256×256 | 19.37 | 0.428 | 0.343 | Grad accum 2 |
| **VDSR** | 667K | 256×256 | 18.75 | 0.446 | 0.285 | Grad accum 2 |
| **Simple CNN** | 1.2K | 256×256 | 17.74 | 0.393 | 0.417 | None |
| **Wiener filter** (baseline) | — | — | **3.56** | **0.007** | **0.832** | CPU only |

> **Key finding:** Learned restoration outperforms classical (Wiener/Richardson-Lucy) by **~18 dB PSNR** — the baseline failure proves learned approaches were the only viable path.

### Physics-Based Degradation Pipeline

Engineered to bridge the gap between synthetic blur and real microscope optics:

| Degradation | Simulation | Parameters |
|-------------|-----------|------------|
| **Defocus** | Gaussian kernel | σ ∈ [0.5, 4.0], kernel sizes 3–21 |
| **Motion blur** | Linear kernel | **30%** probability, variable length + angle |
| **Sensor noise** | Gaussian | σ ∈ [1.0, 6.0] |
| **LED banding** | PWM flicker sinusoid | **50%** probability |
| **Camera ISP** | Gamma correction | γ ∈ [0.7, 1.2] |
| **JPEG compression** | Quality reduction | Q ∈ [50, 95] |

**Domain gap ablation result:** Gaussian-only training drops **7.65 dB** from Gaussian test → real-physics test (25.09 → 17.44 dB). Full-physics training only drops **3.84 dB** (22.33 → 18.49 dB) — a **50% gap reduction**.

### VRAM Optimization (6GB GTX 1660 Ti)

| Technique | Impact |
|-----------|--------|
| **Gradient accumulation** (physical 2 × accum 8 = effective 16) | Enabled SwinIR + ESRGAN on 6GB |
| **Dynamic resolution scaling** (64×64 → 128×128 crops) | SwinIR: fit where OOM otherwise |
| **2GB in-memory RAM cache** | Eliminated SSD I/O bottleneck across 50 epochs |
| **PCIe bottleneck diagnosis** (GPU-Z: 41% bus load) | Batch size reduction → **35→6 min/epoch** |
| **AMP benchmarking** (FP16 vs FP32) | FP32 chosen for biological fidelity |
| **Patch-and-stitch tiling** (25% overlap + alpha blend) | High-res inference on 6GB card |
| **`torch.compile` + Triton fallback** | Graceful on Windows `suppress_errors=True` |

---

## 🧠 Classifier — 3 Backbones

All pre-trained on ImageNet, fine-tuned on restored wood imagery:

| Backbone | Parameters | Frozen Val. Acc | Unfrozen Val. Acc | Live Camera Acc | F1 Score |
|----------|-----------|:---------------:|:-----------------:|:---------------:|:--------:|
| **ResNet18** | 11.2M | 78.34% | **99.93%** | ~86% | 0.971 |
| **ResNet50** | 25.6M | 86.48% | **99.93%** | ~90% | 0.971 |
| **Swin-Tiny** 🏆 | 28.3M | **87.57%** | **99.85%** | **~98%** | **0.975** |

> **Key finding:** All unfrozen backbones converge to 99.85–99.93% validation accuracy (ceiling effect). Swin-T achieves **~98% live camera accuracy** vs ResNet18's ~86% — a **12 pp generalization gap** hidden by validation metrics. **Swin-T is the production choice.**

---

## 📄 Research Paper

| Attribute | Detail |
|-----------|--------|
| Title | *Bridging the Optical Domain Gap: Physics-Based Degradation for Wood Microscopic Image Restoration* |
| Authors | Muhammad Hafiz Hakimi Bin Mohd Zaimi (first author) — CAIRO Lab, UTM |
| Format | Thesis (target: ICME conference format) |
| Length | **14,600 words** draft |
| Artifacts | **4.2 GB** — code, 59 weight files, eval CSVs, figures |
| Key contribution | Quantitative proof that physics-based training halves the synthetic-to-real domain gap (7.65 dB → 3.84 dB) |

---

## 🛠️ Technical Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.13 |
| **GUI Framework** | PyQt6 — multi-threaded, lag-free acquisition interface |
| **Computer Vision** | OpenCV 4.x — VOL sharpness, filter2D, image I/O |
| **Deep Learning** | PyTorch 2.x — 5 architectures, AMP, gradient accumulation |
| **Database** | SQLite3 — metadata tracking, species mapping, classification DB |
| **Eval Framework** | PSNR / SSIM / LPIPS — 50-image holdout with deterministic seed |
| **Hardware** | USB Microscope + Intel i7-10750H + **6GB GTX 1660 Ti** |

---

## ✨ Key Features

### Data Acquisition (PyQt6 GUI)

- **Real-time VOL Metrics** — quantifies sharpness (High VOL > 1000 = Clear) to ensure training data quality
- **Multi-threaded camera pipeline** — eliminates live-microscope lag, sustaining ≥30 FPS
- **Automated Dataset Repair** — filename padding, database-to-disk synchronization
- **Species Registry** — dynamic initials-to-name mapping with safety lock
- **Integrity Verification** — cross-references 6,800+ SQL paths against physical storage

### Training Pipeline

- **5 restoration architectures** — Simple CNN → SwinIR → Real-ESRGAN with one-click training
- **Physics-based degradation** — 8-stage pipeline (defocus, motion, noise, LED, ISP, JPEG, gamma)
- **Gradient accumulation & AMP** — fits 16.7M-param models on 6GB VRAM
- **2GB RAM cache** — eliminates disk I/O across epochs
- **Automatic eval logging** — PSNR/SSIM/LPIPS per epoch with CSV output

### Classifier

- **3-backbone training** — ResNet18/50 + Swin-T with frozen/unfrozen modes
- **Live camera inference** — real-time species classification at ~10ms/image
- **Confusion matrix + F1 reporting** — per-species performance breakdown
- **Restoration comparison mode** — evaluate classifier on each restoration model's output

---

## 📈 Evaluation Results

Standardized 50-image holdout evaluation across all architectures:

```
eval_results/eval_2026-05-06.csv  — Master evaluation (all 5 models × 3 loss variants)
eval_results/eval_2026-05-04.csv  — Early checkpoint evaluation
eval_results/ablation_vdsr_domain_gap.csv  — Domain gap ablation (Gaussian vs physics)
scripts/baseline_results.csv     — Classical baselines (Wiener, Richardson-Lucy)
```

### Quick Reference

| Model | PSNR | SSIM | LPIPS | Best For |
|-------|:----:|:----:|:-----:|----------|
| **SwinIR** | 21.20 dB | 0.473 | 0.272 | **Best fidelity balance** — production choice |
| **Real-ESRGAN** | 21.22 dB | 0.484 | 0.263 | Highest contrast — good for visual inspection |
| **Wiener filter** | 3.56 dB | 0.007 | 0.832 | Classical baseline — catastrophic on wood microscopy |

---

## 🚀 Installation & Usage

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0 (CUDA recommended) |
| OpenCV | ≥ 4.x |
| PyQt6 | ≥ 6.5 |
| GPU | NVIDIA with ≥ 6GB VRAM (GTX 1660 Ti or better) |

### Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd cairo-wood-restoration

# 2. Create virtual environment
python -m venv torch_env
source torch_env/bin/activate  # Linux/WSL
# or: .\torch_env\Scripts\activate  # Windows

# 3. Install PyTorch (CUDA 12.1 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install opencv-python PyQt6 numpy pandas matplotlib scikit-learn seaborn

# 5. Run data acquisition GUI
python app/main.py
```

### Training Workflow

```bash
# Generate physics-based blur dataset
python app/generate_blur_dataset.py

# Train restoration model (e.g., SwinIR)
python app/training_tab.py

# Evaluate restoration
python app/evaluate.py

# Train classifier
python app/train_classifier.py

# Evaluate classifier
python app/evaluate_classifier.py

# Generate restored dataset for classifier training
python app/generate_restored_dataset.py
```

---

## 📁 Project Structure

```
├── app/
│   ├── main.py                          # PyQt6 data acquisition GUI
│   ├── ai.py                            # Acquisition logic + VOL metrics
│   ├── camera_thread.py                 # Multi-threaded camera capture
│   ├── models.py                        # 5 restoration architectures
│   ├── training_tab.py                  # Restoration training UI
│   ├── train.py                         # Training script
│   ├── train_classifier.py              # Classifier training
│   ├── classifier.py                    # Classifier model definitions
│   ├── classifier_training_tab.py       # Classifier training UI
│   ├── recognition_tab.py               # Live classifier inference UI
│   ├── evaluate.py                      # Restoration evaluation (PSNR/SSIM/LPIPS)
│   ├── evaluate_classifier.py           # Classifier evaluation
│   ├── generate_blur_dataset.py         # Physics-based degradation pipeline
│   ├── generate_restored_dataset.py     # Generate restored images for classifier
│   ├── report.py                        # Dataset reporting
│   ├── inspect_weights.py               # Weight file analysis
│   ├── standardize_dataset.py           # Dataset standardization
│   ├── migrate_classification_db.py     # DB migration
│   ├── update_registry.py               # Species registry updater
│   ├── disktodb.py                      # Disk-to-DB sync
│   ├── cleaneup_db.py                   # DB cleanup
│   └── patch_scientific_names.py        # Botanical name patching
├── eval_results/
│   ├── eval_2026-05-06.csv              # Master evaluation (all 5 models)
│   ├── eval_2026-05-04.csv              # Early evaluation
│   └── ablation_vdsr_domain_gap.csv     # Domain gap ablation study
├── scripts/
│   └── baseline_results.csv             # Classical baseline comparisons
├── plans/
│   └── pipeline_assessment.md           # Full pipeline assessment
├── torch_env/                           # Python virtual environment
└── venv/                                # Legacy virtual environment
```

---

## 🔬 Research Findings Summary

| Finding | Detail | Impact |
|---------|--------|--------|
| **Classical restoration fails** | Wiener/RL: PSNR < 4 dB, SSIM < 0.01 | Learned approaches are mandatory for wood microscopy |
| **Physics training halves domain gap** | Full-physics: -3.84 dB vs Gaussian-only: -7.65 dB | **50% reduction** in synthetic-to-real performance drop |
| **SwinIR is production choice** | PSNR 21.20 dB, balanced fidelity, 4.1M params | Best trade-off of quality, speed, and VRAM |
| **Unfrozen backbones converge** | All 3 achieve 99.85–99.93% on validation | Ceiling effect — camera performance is the real discriminator |
| **Swin-T wins on live camera** | ~98% vs ResNet18 ~86% | **12 pp gap** hidden by validation metrics |
| **6GB GPU sufficient** | 16.7M-param ESRGAN trained on consumer card | Gradient accumulation + dynamic scaling + RAM cache enabled this |

---

## 📝 License & Attribution

Developed as part of the **CAIRO Internship Program (2026)** — Centre for Artificial Intelligence & Robotics, Universiti Teknologi Malaysia.

All wood samples provided by the CAIRO Lab. Species identification validated against botanical references.

> *"Bridging the gap between synthetic degradation and real microscope optics — proving that physics-aware training is essential for wood microscopic image restoration."*
