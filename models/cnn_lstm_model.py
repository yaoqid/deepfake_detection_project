"""
CNN-LSTM Hybrid Model for Deepfake Detection
----------------------------------------------
The CNN extracts spatial features from the face image (7x7 = 49 patches).
The LSTM reads these patches as a sequence, capturing long-range spatial
relationships across different face regions.

Why this works for deepfakes:
  - Deepfake artifacts are often LOCAL (one eye slightly off, jawline blur)
  - CNN detects the local artifact in a patch
  - LSTM connects distant patches: "left eye looks fine BUT right eye has
    artifacts AND jawline is blurred → these together = likely fake"

Architecture:
    Face Image (3, 224, 224)
        |
    CNN Backbone (ResNet18, frozen)
        |
    Feature Maps (512, 7, 7) → reshape to (49 patches, 512 features)
        |
    LSTM reads 49 patches as a spatial sequence
        |
    Attention: which face regions matter most?
        |
    Classifier → Real or Fake
"""

import torch
import torch.nn as nn
from torchvision import models


class DeepfakeCNNLSTM(nn.Module):
    """
    CNN feature extractor + LSTM spatial sequence processor.

    The CNN backbone is frozen (no gradient updates) since it's pretrained.
    Only the LSTM and classifier are trained.
    """

    def __init__(
        self,
        feature_dim: int = 512,     # CNN output channels (ResNet18 = 512)
        hidden_size: int = 256,     # LSTM hidden state size
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super(DeepfakeCNNLSTM, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_size = hidden_size

        # ── CNN Feature Extractor (frozen) ────────────────────────────────
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze CNN weights - we only train LSTM
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False

        # ── LSTM Sequence Processor ───────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # ── Spatial Attention ─────────────────────────────────────────────
        # Learns which face patches are most important for detection
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # ── Classifier ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        x: (batch, 3, 224, 224) raw images
        returns: logits (batch, num_classes)
        """
        # Extract spatial features from CNN
        with torch.no_grad():
            feat_maps = self.cnn_backbone(x)  # (batch, 512, 7, 7)

        batch, channels, h, w = feat_maps.shape
        # Reshape to sequence: (batch, 49, 512)
        patches = feat_maps.flatten(2).permute(0, 2, 1)

        # LSTM processes spatial sequence
        lstm_out, _ = self.lstm(patches)  # (batch, 49, hidden*2)

        # Attention: which patches matter most?
        attn_scores = self.attention(lstm_out)            # (batch, 49, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # sum to 1
        context = (attn_weights * lstm_out).sum(dim=1)     # (batch, hidden*2)

        return self.classifier(context)

    def get_attention_map(self, x):
        """
        Returns attention weights reshaped as a 7x7 spatial map.
        Shows which face regions the model focuses on.
        """
        with torch.no_grad():
            feat_maps = self.cnn_backbone(x)

        batch, channels, h, w = feat_maps.shape
        patches = feat_maps.flatten(2).permute(0, 2, 1)

        lstm_out, _ = self.lstm(patches)
        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, 49, 1)

        # Reshape to spatial map: (batch, 7, 7)
        attn_map = attn_weights.squeeze(-1).view(batch, h, w)
        return attn_map


def get_model(num_classes: int = 2, pretrained: bool = True):
    model = DeepfakeCNNLSTM(num_classes=num_classes, pretrained=pretrained)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CNN-LSTM Model: {total:,} total, {trainable:,} trainable "
          f"(CNN frozen: {total - trainable:,})")
    return model


if __name__ == "__main__":
    model = get_model()
    dummy = torch.randn(4, 3, 224, 224)

    output = model(dummy)
    print(f"Input:     {dummy.shape}")
    print(f"Output:    {output.shape}")        # (4, 2)

    attn_map = model.get_attention_map(dummy)
    print(f"Attention: {attn_map.shape}")      # (4, 7, 7)
    print("CNN-LSTM forward pass OK")
