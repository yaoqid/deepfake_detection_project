"""
CNN Model for Deepfake Face Detection
---------------------------------------
Uses pretrained ResNet18 to classify face images as Real or Fake.
Also provides a feature extractor for the CNN-LSTM hybrid model.

Input:  RGB face image (3, 224, 224)
Output: 2 classes (Real / Fake)
"""

import torch
import torch.nn as nn
from torchvision import models


class DeepfakeCNN(nn.Module):
    """
    ResNet18 transfer learning for deepfake detection.

    Architecture:
        Input (3, 224, 224)
            |
        ResNet18 backbone (pretrained)
            |
        Feature maps (512, 7, 7) ← also used by LSTM model
            |
        Global Average Pool → 512
            |
        Classifier: 512 → 256 → 2
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(DeepfakeCNN, self).__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Everything except the final FC layer = feature extractor
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # output: (512, 7, 7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat_maps = self.features(x)       # (batch, 512, 7, 7)
        pooled = self.pool(feat_maps)       # (batch, 512, 1, 1)
        pooled = pooled.flatten(1)          # (batch, 512)
        return self.classifier(pooled)

    def extract_features(self, x):
        """
        Extract spatial feature maps for LSTM input.
        Returns: (batch, 49, 512) — 49 spatial patches, each with 512 features
        """
        with torch.no_grad():
            feat_maps = self.features(x)    # (batch, 512, 7, 7)
        batch, channels, h, w = feat_maps.shape
        # Reshape: (batch, 512, 49) → (batch, 49, 512)
        patches = feat_maps.flatten(2).permute(0, 2, 1)
        return patches

    def get_gradcam(self, x, target_class=1):
        """
        Grad-CAM: highlights which spatial regions drive the prediction
        toward a specific class (default: Fake=1).
        Returns: (batch, 7, 7) heatmap
        """
        self.eval()
        x = x.detach().requires_grad_(False)

        # Must enable gradients even in eval mode
        with torch.enable_grad():
            # Forward through features with gradient tracking
            feat_maps = self.features(x)  # (batch, 512, 7, 7)
            feat_maps.retain_grad()

            pooled = self.pool(feat_maps).flatten(1)
            logits = self.classifier(pooled)

            # Backward from target class
            logits[:, target_class].sum().backward()

        # Grad-CAM: weight each channel by its gradient, then ReLU
        grads = feat_maps.grad  # (batch, 512, 7, 7)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (batch, 512, 1, 1)
        cam = (weights * feat_maps).sum(dim=1)  # (batch, 7, 7)
        cam = torch.relu(cam)  # only positive contributions

        # Normalize per sample
        cam = cam.detach()
        batch_size = cam.shape[0]
        for i in range(batch_size):
            cam_min = cam[i].min()
            cam_max = cam[i].max()
            if cam_max - cam_min > 0:
                cam[i] = (cam[i] - cam_min) / (cam_max - cam_min)
            else:
                cam[i] = torch.zeros_like(cam[i])

        return cam


def get_model(num_classes: int = 2, pretrained: bool = True):
    model = DeepfakeCNN(num_classes=num_classes, pretrained=pretrained)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CNN Model: {total:,} total params, {trainable:,} trainable")
    return model


if __name__ == "__main__":
    model = get_model()
    dummy = torch.randn(4, 3, 224, 224)

    output = model(dummy)
    print(f"Input:    {dummy.shape}")
    print(f"Output:   {output.shape}")       # (4, 2)

    features = model.extract_features(dummy)
    print(f"Features: {features.shape}")     # (4, 49, 512)
    print("CNN forward pass OK")
