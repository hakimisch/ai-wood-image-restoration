# app/classifier.py
#
# Species Classification Module
# Uses transfer learning (ResNet18) to identify wood species from microscope images.
# Designed for real-time inference on GTX 1660 Ti (6GB VRAM).
#
# Usage:
#   from classifier import SpeciesClassifier
#   clf = SpeciesClassifier(num_species=35)
#   clf.load_weights("classifier_weights.pth")
#   species_name, confidence, top3 = clf.predict(img_bgr)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import numpy as np
import os
import json

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Species Name ↔ Index Mapping
# ---------------------------------------------------------------------------
def build_species_index(db_path="data/database.db"):
    """Reads species_registry from DB and returns (idx_to_name, name_to_idx)."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT initials, full_name FROM species_registry ORDER BY initials")
    rows = cursor.fetchall()
    conn.close()

    # Deduplicate by full_name
    seen = set()
    unique = []
    for initials, name in rows:
        if name not in seen:
            seen.add(name)
            unique.append((initials, name))

    idx_to_name = {i: name for i, (_, name) in enumerate(unique)}
    name_to_idx = {name: i for i, name in idx_to_name.items()}
    return idx_to_name, name_to_idx


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SpeciesClassifier(nn.Module):
    """ResNet18-based wood species classifier.

    Args:
        num_species: Number of wood species to classify.
        freeze_backbone: If True, only the classifier head is trainable (transfer learning).
    """

    def __init__(self, num_species=35, freeze_backbone=True):
        super().__init__()

        # Load pretrained ResNet18
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Freeze backbone weights if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final fully-connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_species),
        )

        self.num_species = num_species

        # Input normalization (ImageNet stats)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        """x: (B, 3, H, W) float32 tensor, values in [0, 1]."""
        return self.backbone(x)

    @torch.no_grad()
    def predict(self, img_bgr, top_k=3):
        """Run inference on a single BGR image.

        Args:
            img_bgr: (H, W, 3) uint8 BGR numpy array.
            top_k: Number of top predictions to return.

        Returns:
            (species_name, confidence, top3_list)
            where top3_list is [(species_name, confidence), ...]
        """
        self.eval()
        self.to(_DEVICE)

        # Preprocess
        tensor = self._preprocess(img_bgr).unsqueeze(0).to(_DEVICE)

        # Inference
        logits = self(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        # Top-k
        values, indices = torch.topk(probs, k=min(top_k, self.num_species))
        values = values.cpu().numpy()
        indices = indices.cpu().numpy()

        top3 = [(self.idx_to_name[int(i)], float(v)) for i, v in zip(indices, values)]
        return top3[0][0], top3[0][1], top3

    def _preprocess(self, img_bgr):
        """Convert BGR uint8 → normalized CHW tensor."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (224, 224))
        tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        return self.normalize(tensor)

    def load_weights(self, path, map_location=None):
        """Load trained weights from a .pth file."""
        if map_location is None:
            map_location = _DEVICE
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state_dict, strict=False)
        self.eval()
        print(f"✅ Classifier weights loaded: {path}")

    def save_weights(self, path):
        """Save model weights to a .pth file."""
        torch.save(self.state_dict(), path)
        print(f"💾 Classifier weights saved: {path}")


# ---------------------------------------------------------------------------
# Convenience: build a classifier ready for inference
# ---------------------------------------------------------------------------

def create_classifier(weights_path=None, db_path="data/database.db"):
    """Factory function: builds SpeciesClassifier, loads index mapping, optionally loads weights."""
    idx_to_name, name_to_idx = build_species_index(db_path)
    model = SpeciesClassifier(num_species=len(idx_to_name))
    model.idx_to_name = idx_to_name
    model.name_to_idx = name_to_idx

    if weights_path and os.path.exists(weights_path):
        model.load_weights(weights_path)

    return model
