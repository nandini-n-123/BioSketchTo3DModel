from __future__ import annotations

import torch.nn as nn

SUPPORTED_MODELS = {"efficientnet_b0", "mobilenet_v3_small", "resnet18"}


def _set_requires_grad(model: nn.Module, value: bool) -> None:
    for p in model.parameters():
        p.requires_grad = value


def create_model(
    model_name: str,
    num_classes: int,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Create a torchvision classification model with a 3-class head."""
    model_name = model_name.lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_name={model_name}. Choose from {sorted(SUPPORTED_MODELS)}")

    if model_name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        if freeze_backbone:
            _set_requires_grad(model, False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        for p in model.classifier.parameters():
            p.requires_grad = True
        return model

    if model_name == "mobilenet_v3_small":
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        if freeze_backbone:
            _set_requires_grad(model, False)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        for p in model.classifier.parameters():
            p.requires_grad = True
        return model

    if model_name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        if freeze_backbone:
            _set_requires_grad(model, False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        for p in model.fc.parameters():
            p.requires_grad = True
        return model

    raise AssertionError("unreachable")


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True
