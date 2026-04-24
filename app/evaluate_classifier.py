# app/evaluate_classifier.py
#
# Standalone evaluation script for the Species Classifier.
# Loads trained weights, runs on the holdout test set, and generates:
#   - Per-species precision, recall, F1-score
#   - Confusion matrix (saved as PNG)
#   - Accuracy bar chart
#   - Console report
#
# Usage:
#   python app/evaluate_classifier.py --weights classifier_weights.pth

import os
import sys
import argparse
import json
from datetime import datetime
from collections import defaultdict

import numpy as np
import cv2
import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.classifier import create_classifier
from app.train_classifier import WoodClassificationDataset

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ matplotlib not installed. Charts will be skipped.")

try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not installed. Install with: pip install scikit-learn")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Evaluating on device: {device}")
    print(f"{'='*60}")

    # Load classifier
    if not os.path.exists(args.weights):
        print(f"❌ Weights not found: {args.weights}")
        return

    classifier = create_classifier(weights_path=args.weights)
    classifier = classifier.to(device)
    classifier.eval()
    print(f"✅ Classifier loaded: {classifier.num_species} species")

    # Load validation dataset
    val_dataset = WoodClassificationDataset(
        db_path='data/database.db',
        kayu_dir='Kayu',
        split="val",
        val_ratio=args.val_ratio,
        seed=42,
        augment=False,
    )
    print(f"📊 Validation samples: {len(val_dataset)}")

    # Run inference on all validation samples
    all_preds = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for i in range(len(val_dataset)):
            tensor, label = val_dataset[i]
            tensor = tensor.unsqueeze(0).to(device)

            logits = classifier(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()

            all_preds.append(pred)
            all_labels.append(label)
            all_confidences.append(confidence)

            if (i + 1) % 100 == 0:
                print(f"  Evaluated {i+1}/{len(val_dataset)}...")

    # ── Metrics ───────────────────────────────────────────────────────────

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n{'='*60}")
    print(f"📈 OVERALL ACCURACY: {accuracy:.2%}")
    print(f"{'='*60}")

    # Per-species report
    target_names = [classifier.idx_to_name[i] for i in range(classifier.num_species)]
    report = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        digits=4,
    )
    print("\n📋 Per-Species Classification Report:")
    print(report)

    # Save report to file
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(f"Classification Report\n")
        f.write(f"Weights: {args.weights}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n\n")
        f.write(report)
    print(f"💾 Report saved to: {report_path}")

    # ── Confusion Matrix ──────────────────────────────────────────────────

    if MATPLOTLIB_AVAILABLE:
        cm = confusion_matrix(all_labels, all_preds)

        # Normalize
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

        # Show values in cells
        num_species = len(target_names)
        for i in range(num_species):
            for j in range(num_species):
                val = cm_norm[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

        ax.set_xticks(range(num_species))
        ax.set_yticks(range(num_species))
        ax.set_xticklabels(target_names, rotation=90, fontsize=8)
        ax.set_yticklabels(target_names, fontsize=8)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.2%})', fontsize=14)

        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()

        cm_path = f"reports/confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"💾 Confusion matrix saved to: {cm_path}")

        # ── Accuracy Bar Chart ────────────────────────────────────────────

        per_species_acc = cm_norm.diagonal()
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        colors = ['#27ae60' if a >= 0.8 else '#f39c12' if a >= 0.5 else '#e74c3c'
                  for a in per_species_acc]
        bars = ax2.bar(range(num_species), per_species_acc, color=colors)
        ax2.set_xticks(range(num_species))
        ax2.set_xticklabels(target_names, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Per-Species Classification Accuracy', fontsize=14)
        ax2.set_ylim(0, 1.05)
        ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80% target')
        ax2.legend()

        # Add value labels on bars
        for bar, acc in zip(bars, per_species_acc):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{acc:.1%}', ha='center', va='bottom', fontsize=7)

        fig2.tight_layout()
        chart_path = f"reports/per_species_accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig2.savefig(chart_path, dpi=150)
        plt.close(fig2)
        print(f"💾 Accuracy chart saved to: {chart_path}")

    print(f"\n{'='*60}")
    print(f"✅ Evaluation complete!")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the species classifier on holdout validation set."
    )
    parser.add_argument("--weights", "-w", type=str, default="classifier_weights.pth",
                        help="Path to trained classifier weights (default: classifier_weights.pth)")
    parser.add_argument("--val_ratio", "-v", type=float, default=0.2,
                        help="Validation split ratio (must match training, default: 0.2)")
    args = parser.parse_args()

    if not SKLEARN_AVAILABLE:
        print("❌ scikit-learn is required for evaluation metrics.")
        print("   Install: pip install scikit-learn")
        return

    evaluate(args)


if __name__ == "__main__":
    main()
