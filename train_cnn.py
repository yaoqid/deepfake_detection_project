"""
Train the CNN Model - Deepfake Detection
------------------------------------------
Usage:
    python train_cnn.py                                  # synthetic data
    python train_cnn.py --data-dir data/Dataset          # real data
    python train_cnn.py --data-dir data/Dataset --epochs 10 --max-per-class 5000
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.cnn_model import get_model
from data.data_loader import get_cnn_dataloaders, NUM_CLASSES


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Validating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def train(args):
    device = get_best_device()
    print(f"Training on: {device}")
    os.makedirs("checkpoints", exist_ok=True)

    print("\nLoading data...")
    train_loader, val_loader = get_cnn_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        n_synthetic=args.n_samples,
        max_per_class=args.max_per_class,
    )

    print("\nBuilding CNN model...")
    model = get_model(num_classes=NUM_CLASSES, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    print(f"\nStarting training for {args.epochs} epochs...\n")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/cnn_best.pt")
            print(f"  New best! Val Acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}\n")

    with open("checkpoints/cnn_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best Val Accuracy: {best_val_acc:.4f}")
    print("Model saved to: checkpoints/cnn_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake CNN Detector")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Limit images per class (for faster testing)")
    train(parser.parse_args())
