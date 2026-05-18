from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from biosketch_classifier.data import ManifestDataset
from biosketch_classifier.models import create_model, unfreeze_all
from biosketch_classifier.preprocessing import make_transforms
from biosketch_classifier.utils import inverse_frequency_class_weights, load_json, sample_weights_from_targets, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BioSketch organ classifier")
    parser.add_argument("--dataset-root", default="data/raw/datasets")
    parser.add_argument("--manifest", default="data/splits/manifest.csv")
    parser.add_argument("--classes", default="data/splits/classes.json")
    parser.add_argument("--out-dir", default="checkpoints/organ_classifier")
    parser.add_argument("--model", default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_small", "resnet18"])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--freeze-epochs", type=int, default=5, help="Train only classifier head for this many epochs, then fine-tune all layers")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weighted-sampler", action=argparse.BooleanOptionalAction, default=False, help="Oversample minority classes in training batches")
    parser.add_argument("--scan-preprocess-train", action=argparse.BooleanOptionalAction, default=False, help="Apply Adobe-Scan-like thresholding to training/validation/test dataset images")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use mixed precision on CUDA")
    return parser.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)
    all_preds, all_targets = [], []
    total_loss = 0.0

    pbar = tqdm(loader, leave=False, desc="train" if is_train else "eval")
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if scaler is not None and scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        all_preds.extend(preds)
        all_targets.extend(y.detach().cpu().numpy().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, bal_acc, macro_f1, np.array(all_targets), np.array(all_preds)


def make_loader(dataset, batch_size: int, num_workers: int, train: bool, weighted_sampler: bool, num_classes: int):
    if train and weighted_sampler:
        weights = sample_weights_from_targets(dataset.targets, num_classes)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_to_idx = load_json(args.classes)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)

    train_ds = ManifestDataset(
        args.dataset_root,
        args.manifest,
        class_to_idx,
        split="train",
        transform=make_transforms(train=True, image_size=args.image_size),
        scan_preprocess=args.scan_preprocess_train,
    )
    val_ds = ManifestDataset(
        args.dataset_root,
        args.manifest,
        class_to_idx,
        split="val",
        transform=make_transforms(train=False, image_size=args.image_size),
        scan_preprocess=args.scan_preprocess_train,
    )
    test_ds = ManifestDataset(
        args.dataset_root,
        args.manifest,
        class_to_idx,
        split="test",
        transform=make_transforms(train=False, image_size=args.image_size),
        scan_preprocess=args.scan_preprocess_train,
    )

    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, True, args.weighted_sampler, num_classes)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers, False, False, num_classes)
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers, False, False, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {class_to_idx}")
    print(f"Split sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    model = create_model(args.model, num_classes, pretrained=True, freeze_backbone=(args.freeze_epochs > 0)).to(device)

    if args.use_class_weights:
        weights = inverse_frequency_class_weights(train_ds.targets, num_classes).to(device)
        print(f"Class weights: {weights.detach().cpu().numpy().round(3).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr_head, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_bal_acc = -1.0
    best_path = out_dir / "best_model.pt"
    epochs_without_improve = 0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("Unfreezing backbone for fine-tuning...")
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune, weight_decay=args.weight_decay)

        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc, train_bal, train_f1, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_acc, val_bal, val_f1, val_targets, val_preds = run_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_balanced_acc": train_bal,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_balanced_acc": val_bal,
            "val_macro_f1": val_f1,
        }
        history_rows.append(row)
        print(
            f"train loss={train_loss:.4f} acc={train_acc:.3f} bal_acc={train_bal:.3f} f1={train_f1:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} bal_acc={val_bal:.3f} f1={val_f1:.3f}"
        )

        if val_bal > best_val_bal_acc:
            best_val_bal_acc = val_bal
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "image_size": args.image_size,
                    "scan_preprocess_train": args.scan_preprocess_train,
                    "val_balanced_acc": val_bal,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"Saved best model -> {best_path}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f"Early stopping after {args.patience} epochs without val balanced-accuracy improvement.")
                break

    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)
    print(f"Wrote history -> {history_path}")

    print("\nEvaluating best checkpoint on test split...")
    checkpoint = torch.load(best_path, map_location=device)
    model = create_model(checkpoint["model_name"], num_classes, pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, test_bal, test_f1, test_targets, test_preds = run_epoch(model, test_loader, criterion, device)

    report = classification_report(test_targets, test_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(test_targets, test_preds, labels=list(range(num_classes)))
    print(report)
    print("Confusion matrix rows=true, cols=pred:")
    print(cm)

    (out_dir / "test_report.txt").write_text(
        f"test_loss={test_loss:.6f}\ntest_acc={test_acc:.6f}\ntest_balanced_acc={test_bal:.6f}\ntest_macro_f1={test_f1:.6f}\n\n{report}\n\nConfusion matrix rows=true cols=pred\n{cm}\n",
        encoding="utf-8",
    )
    np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    save_json(vars(args), out_dir / "train_args.json")
    print(f"Best checkpoint: {best_path.resolve()}")


if __name__ == "__main__":
    main()
