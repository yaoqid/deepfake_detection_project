"""
Data Loading & Preprocessing for Deepfake Detection
------------------------------------------------------
Loads real and fake face images for:
  1. CNN training   - individual frame classification (Real vs Fake)
  2. CNN-LSTM model - spatial patch sequence analysis

Dataset: Deepfake and Real Images (Kaggle)
  Layout: Dataset/Train/Real/, Dataset/Train/Fake/, etc.
  ~140K train, ~40K val, ~11K test images
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ── Class definitions ─────────────────────────────────────────────────────────
CLASS_NAMES = ["Real", "Fake"]
NUM_CLASSES = 2
IMAGE_SIZE = 224
PATCH_GRID = 7  # 7x7 = 49 spatial patches for LSTM


# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


# ── Dataset: Folder-based face images ─────────────────────────────────────────
class DeepfakeDataset(Dataset):
    """
    Loads face images from folder structure:
        root/Real/*.jpg  → label 0
        root/Fake/*.jpg  → label 1
    """

    def __init__(self, root_dir: str, transform=None, max_per_class: int = None):
        self.transform = transform or val_transform
        self.image_paths = []
        self.labels = []

        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"  Warning: {class_dir} not found")
                continue

            files = sorted([f for f in os.listdir(class_dir)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))])

            if max_per_class:
                files = files[:max_per_class]

            for fname in files:
                self.image_paths.append(os.path.join(class_dir, fname))
                self.labels.append(class_idx)

        print(f"  Loaded {len(self.image_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


# ── Synthetic dataset for testing without download ────────────────────────────
class SyntheticFaceDataset(Dataset):
    """Generates simple synthetic face-like images for pipeline testing."""

    def __init__(self, n_samples: int = 500, transform=None):
        self.n_samples = n_samples
        self.transform = transform or val_transform
        self.labels = np.random.randint(0, NUM_CLASSES, n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        label = self.labels[idx]

        # Create a simple face-like image
        skin = np.random.randint(170, 220)
        img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), skin, dtype=np.uint8)

        # Face oval
        cy, cx = IMAGE_SIZE // 2, IMAGE_SIZE // 2
        y_g, x_g = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
        face_mask = ((x_g - cx) / 70) ** 2 + ((y_g - cy) / 90) ** 2 <= 1
        face_color = np.array([skin - 20, skin - 10, skin], dtype=np.uint8)
        img[face_mask] = face_color

        if label == 1:  # Fake - add artifacts
            # Blurring artifact on one side
            img[:, IMAGE_SIZE // 2:] = (img[:, IMAGE_SIZE // 2:] * 0.8).astype(np.uint8)
            # Slight colour shift
            img[:, :, 0] = np.clip(img[:, :, 0].astype(int) + 15, 0, 255).astype(np.uint8)

        noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(img)
        if self.transform:
            pil_img = self.transform(pil_img)

        return pil_img, torch.tensor(label, dtype=torch.long)


# ── DataLoader helpers ────────────────────────────────────────────────────────
def _find_dataset_root(data_dir: str) -> str:
    """Find the root folder containing Train/Test/Validation."""
    if os.path.isdir(os.path.join(data_dir, "Train")):
        return data_dir

    for item in os.listdir(data_dir):
        sub = os.path.join(data_dir, item)
        if os.path.isdir(sub) and os.path.isdir(os.path.join(sub, "Train")):
            return sub

    return data_dir


def get_cnn_dataloaders(data_dir: str = None, batch_size: int = 32,
                        n_synthetic: int = 1000, max_per_class: int = None):
    """
    Returns train/val/test DataLoaders for CNN.
    Uses real data if data_dir exists, otherwise synthetic.
    """
    if data_dir and os.path.exists(data_dir):
        root = _find_dataset_root(data_dir)
        train_dir = os.path.join(root, "Train")
        val_dir = os.path.join(root, "Validation")
        test_dir = os.path.join(root, "Test")

        if os.path.isdir(train_dir):
            print("Loading deepfake dataset...")
            train_ds = DeepfakeDataset(train_dir, train_transform, max_per_class)
            val_ds = DeepfakeDataset(val_dir, val_transform, max_per_class)

            if len(train_ds) == 0:
                print("No images found, using synthetic data")
                return _synthetic_loaders(n_synthetic, batch_size)

            train_loader = DataLoader(train_ds, batch_size=batch_size,
                                      shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size,
                                    shuffle=False, num_workers=4, pin_memory=True)
            return train_loader, val_loader
        else:
            print(f"Train folder not found in {root}")

    return _synthetic_loaders(n_synthetic, batch_size)


def _synthetic_loaders(n_samples, batch_size):
    """Create synthetic DataLoaders for testing."""
    n_val = int(n_samples * 0.2)
    n_train = n_samples - n_val

    train_ds = SyntheticFaceDataset(n_train, train_transform)
    val_ds = SyntheticFaceDataset(n_val, val_transform)
    print(f"Using synthetic data: {n_train} train / {n_val} val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def get_test_loader(data_dir: str, batch_size: int = 32):
    """Load test set for final evaluation."""
    root = _find_dataset_root(data_dir)
    test_dir = os.path.join(root, "Test")
    test_ds = DeepfakeDataset(test_dir, val_transform)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)


# ── Video Processing ─────────────────────────────────────────────────────────
def extract_frames_from_video(video_path: str, max_frames: int = 30,
                               sample_rate: int = None) -> list:
    """
    Extract frames from a video file.

    Args:
        video_path: path to video file (.mp4, .avi, .mov, etc.)
        max_frames: maximum number of frames to extract
        sample_rate: extract every Nth frame. If None, evenly samples max_frames.

    Returns:
        List of PIL.Image frames
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        raise ValueError("Could not read frame count from video")

    # Determine which frames to extract
    if sample_rate:
        frame_indices = list(range(0, total_frames, sample_rate))[:max_frames]
    else:
        # Evenly sample max_frames across the video
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # OpenCV uses BGR, convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)

    cap.release()

    return frames, fps, total_frames


def get_video_info(video_path: str) -> dict:
    """Get basic video metadata."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing synthetic data...")
    train_loader, val_loader = get_cnn_dataloaders(batch_size=8)
    imgs, labels = next(iter(train_loader))
    print(f"Batch - images: {imgs.shape}, labels: {labels.shape}")
    print("Data pipeline OK")
