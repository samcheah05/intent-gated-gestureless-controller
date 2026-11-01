import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model import TemporalGestureCNN, GESTURE_CLASSES

def generate_synthetic_gesture_dataset(samples_per_class: int = 250, seq_len: int = 30):
    """
    Generates synthetic normalized 21-keypoint kinematic trajectories for the 5 gesture classes
    so the model can be trained, evaluated, and exported immediately for live demo.
    """
    X, y = [], []
    rng = np.random.default_rng(42)

    # Base hand skeleton template (21 joints x 3 coords)
    base_skeleton = rng.uniform(-0.3, 0.3, size=(21, 3)).astype(np.float32)
    base_skeleton[0] = [0.0, 0.0, 0.0]   # Wrist anchor
    base_skeleton[9] = [0.0, -1.0, 0.0]  # Middle MCP unit scale

    for class_idx in range(len(GESTURE_CLASSES)):
        for _ in range(samples_per_class):
            seq = np.zeros((seq_len, 21, 3), dtype=np.float32)
            speed = rng.uniform(0.6, 1.4)

            for t in range(seq_len):
                progress = (t / float(seq_len - 1)) * speed
                frame_skel = base_skeleton.copy() + rng.normal(0, 0.015, size=(21, 3)).astype(np.float32)

                # Apply directional kinematics mainly on extended fingertips (landmarks 4, 8, 12, 16, 20)
                if class_idx == 1:    # SWIPE_LEFT
                    frame_skel[5:, 0] -= progress * 1.5
                elif class_idx == 2:  # SWIPE_RIGHT
                    frame_skel[5:, 0] += progress * 1.5
                elif class_idx == 3:  # SWIPE_UP
                    frame_skel[5:, 1] -= progress * 1.5
                elif class_idx == 4:  # SWIPE_DOWN
                    frame_skel[5:, 1] += progress * 1.5

                # Re-anchor wrist
                frame_skel[0] = [0.0, 0.0, 0.0]
                seq[t] = frame_skel

            X.append(seq.reshape(seq_len, 63))
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Shuffle dataset
    indices = rng.permutation(len(X))
    return X[indices], y[indices]


def train_and_export(weights_path: str = "weights/gesture_1dcnn.pth", epochs: int = 20):
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PyTorch] Training TemporalGestureCNN on device: {device}")

    X, y = generate_synthetic_gesture_dataset(samples_per_class=300, seq_len=30)
    split = int(0.8 * len(X))

    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    val_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = TemporalGestureCNN(num_features=63, num_classes=len(GESTURE_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x).argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        acc = (correct / total) * 100.0
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Loss: {train_loss/len(train_loader):.4f} | Val Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), weights_path)
    print(f"[PyTorch] Model saved successfully to {weights_path}")


if __name__ == "__main__":
    train_and_export()
