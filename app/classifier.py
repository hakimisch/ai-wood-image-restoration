# app/classifier.py
#
# Species Classification Module
# Uses transfer learning (ResNet18, ResNet50, Swin-Tiny) to identify wood species
# from USB microscope images. Designed for real-time inference on GTX 1660 Ti (6GB VRAM).
#
# Usage:
#   from classifier import SpeciesClassifier
#   clf = SpeciesClassifier(num_species=35, backbone_name='swin_t')
#   clf.load_weights("classifier_weights.pth")
#   species_name, confidence, top3 = clf.predict(img_bgr)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    swin_t, Swin_T_Weights,
)
import cv2
import numpy as np
import os

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Backbone Registry
# ---------------------------------------------------------------------------
# Maps backbone name → (builder_fn, weights_enum, classifier_attr, in_features)
# classifier_attr: the attribute name on the model that is the classification head
_BACKBONES = {
    'resnet18': {
        'builder': resnet18,
        'weights_enum': ResNet18_Weights.IMAGENET1K_V1,
        'classifier_attr': 'fc',
        'in_features': 512,
    },
    'resnet50': {
        'builder': resnet50,
        'weights_enum': ResNet50_Weights.IMAGENET1K_V1,
        'classifier_attr': 'fc',
        'in_features': 2048,
    },
    'swin_t': {
        'builder': swin_t,
        'weights_enum': Swin_T_Weights.IMAGENET1K_V1,
        'classifier_attr': 'head',
        'in_features': 768,
    },
}

VALID_BACKBONES = set(_BACKBONES.keys())


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
# Weight metadata helpers
# ---------------------------------------------------------------------------
_CLASSIFIER_META_KEY = '_classifier_meta'


def _build_default_head(in_features, num_species):
    """Build the standard classifier head used across all backbones."""
    return nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_species),
    )


def detect_backbone_from_weights(path):
    """Read backbone_name embedded in a saved .pth file, or default to 'resnet18'.

    New-format weights (saved by SpeciesClassifier.save_weights) include a
    '_classifier_meta' key. Old-format weights are assumed to be ResNet18.
    """
    if not os.path.exists(path):
        return 'resnet18'
    try:
        data = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(data, dict) and _CLASSIFIER_META_KEY in data:
            return data[_CLASSIFIER_META_KEY].get('backbone_name', 'resnet18')
    except Exception:
        pass
    return 'resnet18'


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SpeciesClassifier(nn.Module):
    """Multi-backbone wood species classifier.

    Args:
        num_species: Number of wood species to classify.
        freeze_backbone: If True, only the classifier head is trainable (transfer learning).
        backbone_name: One of 'resnet18', 'resnet50', 'swin_t'.
    """

    def __init__(self, num_species=35, freeze_backbone=True, backbone_name='resnet18'):
        super().__init__()

        if backbone_name not in _BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Valid options: {sorted(VALID_BACKBONES)}"
            )

        self.backbone_name = backbone_name
        info = _BACKBONES[backbone_name]

        # Load pretrained backbone
        backbone = info['builder'](weights=info['weights_enum'])

        # Freeze backbone weights if requested
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # Replace the classifier head with our custom head
        classifier_attr = info['classifier_attr']
        new_head = _build_default_head(info['in_features'], num_species)
        setattr(backbone, classifier_attr, new_head)

        self.backbone = backbone
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

    def load_weights(self, path, map_location=None, verbose=True):
        """Load trained weights from a .pth file.

        Supports both new-format (with '_classifier_meta' key) and old-format
        (bare state_dict) weight files. Uses strict=False to handle
        classifier-head dimension mismatches gracefully.
        """
        if map_location is None:
            map_location = _DEVICE
        data = torch.load(path, map_location=map_location, weights_only=True)

        # Check if saved with metadata wrapper
        if isinstance(data, dict) and _CLASSIFIER_META_KEY in data:
            state_dict = data['model_state_dict']
        else:
            state_dict = data

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        self.eval()

        if verbose:
            info_parts = [f"✅ Classifier weights loaded: {path}"]
            if missing:
                info_parts.append(f"  Missing keys (head mismatch): {len(missing)}")
            if unexpected:
                info_parts.append(f"  Unexpected keys: {len(unexpected)}")
            print("\n".join(info_parts))

        return missing, unexpected

    def save_weights(self, path):
        """Save model weights to a .pth file (includes metadata)."""
        torch.save({
            'model_state_dict': self.state_dict(),
            _CLASSIFIER_META_KEY: {
                'backbone_name': self.backbone_name,
                'num_species': self.num_species,
            },
        }, path)
        print(f"💾 Classifier weights saved: {path}")


# ---------------------------------------------------------------------------
# Convenience: build a classifier ready for inference
# ---------------------------------------------------------------------------

def create_classifier(weights_path=None, db_path="data/database.db",
                      backbone_name=None):
    """Factory function: builds SpeciesClassifier, loads index mapping,
    optionally loads weights.

    Args:
        weights_path: Path to .pth weights file. If None, returns untrained model.
        db_path: Path to SQLite database for species registry.
        backbone_name: One of 'resnet18', 'resnet50', 'swin_t'.
            If None, auto-detects from weights_path metadata or defaults to 'resnet18'.

    Returns:
        SpeciesClassifier instance with idx_to_name and name_to_idx mappings.
    """
    # Resolve backbone_name
    if backbone_name is None:
        if weights_path and os.path.exists(weights_path):
            backbone_name = detect_backbone_from_weights(weights_path)
        else:
            backbone_name = 'resnet18'

    idx_to_name, name_to_idx = build_species_index(db_path)
    num_species = len(idx_to_name)
    model = SpeciesClassifier(
        num_species=num_species,
        freeze_backbone=False,  # no effect; weights will be loaded
        backbone_name=backbone_name,
    )
    model.idx_to_name = idx_to_name
    model.name_to_idx = name_to_idx

    if weights_path and os.path.exists(weights_path):
        model.load_weights(weights_path)

    return model
