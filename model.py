import torch
import torch.nn as nn
import numpy as np

GESTURE_CLASSES = [
    "IDLE",
    "SWIPE_LEFT",
    "SWIPE_RIGHT",
    "SWIPE_UP",
    "SWIPE_DOWN"
]

class TemporalGestureCNN(nn.Module):
    """
    PyTorch 1D-CNN Temporal Gesture Classifier (Section 7.1.5)
    Input Shape: [Batch_Size, Sequence_Length=30, Features=63]
    """
    def __init__(self, num_features: int = 63, num_classes: int = 5):
        super(TemporalGestureCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Temporal 30 -> 15

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.dropout = nn.Dropout(p=0.3)
        self.fc_out = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose from [Batch, Seq_Len=30, Features=63] -> [Batch, Features=63, Seq_Len=30]
        x = x.transpose(1, 2)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.global_pool(self.relu(self.bn2(self.conv2(x))))
        x = x.squeeze(-1)  # [Batch, 256]
        x = self.dropout(self.relu(self.fc1(x)))
        logits = self.fc_out(x)
        return logits


def normalize_landmarks(landmark_list) -> np.ndarray:
    """
    Spatial Coordinate Normalization (Section 7.1.2)
    Centers 21 3D landmarks at Wrist (L_0) and scales by Palm Size ||L_9 - L_0||_2.
    Returns flat 63-element float32 vector.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmark_list], dtype=np.float32)
    wrist = coords[0]
    middle_mcp = coords[9]
    s_hand = np.linalg.norm(middle_mcp - wrist)
    if s_hand < 1e-6:
        s_hand = 1.0
    normalized = (coords - wrist) / s_hand
    return normalized.flatten()
