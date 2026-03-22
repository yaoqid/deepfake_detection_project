"""
Train the CNN-LSTM Model - Deepfake Spatial Analysis
------------------------------------------------------
The CNN backbone extracts features, LSTM processes them as spatial sequences.
Only the LSTM + classifier weights are trained (CNN is frozen).

Usage:
    python train_lstm.py                                    # synthetic data
    python train_lstm.py --data-dir data/Dataset            # real data
    python train_lstm.py --data-dir data/Dataset --max-per-class 5000
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.cnn_lstm_model import get_model
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
        outputs = model(images)  # CNN-LSTM takes raw images
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

    for images, labels in loader:
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

    print("\nBuilding CNN-LSTM model...")
    model = get_model(num_classes=NUM_CLASSES, pretrained=True).to(device)

    # Only train LSTM parameters (CNN backbone is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nStarting CNN-LSTM training for {args.epochs} epochs...\n")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/cnn_lstm_best.pt")
            print(f"  New best! Val Acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}\n")

    with open("checkpoints/cnn_lstm_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nCNN-LSTM Training complete! Best Val Accuracy: {best_val_acc:.4f}")
    print("Model saved to: checkpoints/cnn_lstm_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake CNN-LSTM Detector")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--max-per-class", type=int, default=None)
    train(parser.parse_args())
