from __future__ import annotations

# Folder names in your current datasets.zip mapped to final backend labels.
CLASS_FOLDER_ALIASES = {
    "Brain_Dataset": "brain",
    "Heart_Dataset": "heart",
    "Lungs_Dataset": "lungs",
    "Brain": "brain",
    "Heart": "heart",
    "Lungs": "lungs",
    "brain": "brain",
    "heart": "heart",
    "lungs": "lungs",
}

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ImageNet normalization used by torchvision pretrained CNN weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
