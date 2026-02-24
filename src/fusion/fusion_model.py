import torch
import torch.nn as nn


class FusionModel(nn.Module):
    """
    Fusion model with dual modes:
    1. Text-only (training)
    2. Text + Visual (inference / fusion)
    """

    def __init__(
        self,
        visual_dim=1280,
        text_dim=768,
        hidden_dim=512,
        num_classes=3
    ):
        super().__init__()

        # Optional visual branch
        self.fc_visual = nn.Linear(visual_dim, hidden_dim)

        # Text branch (always used)
        self.fc_text = nn.Linear(text_dim, hidden_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        # Text-only classifier (NEW)
        self.text_only_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, text_feat, visual_feat=None):
        """
        text_feat: (B, 768)
        visual_feat: (B, 1280) or None
        """

        t = self.relu(self.fc_text(text_feat))
        t = self.dropout(t)

        # 🔹 TEXT-ONLY MODE (TRAINING)
        if visual_feat is None:
            return self.text_only_classifier(t)

        # 🔹 FUSION MODE (INFERENCE)
        v = self.relu(self.fc_visual(visual_feat))
        v = self.dropout(v)

        combined = torch.cat((t, v), dim=1)
        return self.classifier(combined)
