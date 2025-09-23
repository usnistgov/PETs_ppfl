from typing import Dict
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, features: int):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 12, kernel_size=14)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(12, 10, kernel_size=10)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.conv3 = nn.Conv1d(10, 8, kernel_size=8)
        self.relu3 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.batch_norm1 = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(
            in_features=self._calculate_flatten_size(features), out_features=48
        )
        self.fc2 = nn.Linear(in_features=48, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=5)

    def _calculate_flatten_size(self, features: int):
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.zeros(1, 1, features).to(device)
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.pool(x)
            x = self.batch_norm1(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(
                next(self.parameters()).device
            )
        x = x.float()
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool(x)
        x = self.batch_norm1(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


def train_cnn(
    model_id: int,
    train_loader,
    test_loader,
    epochs,
    model,
    optimizer,
    criterion,
    device="cpu",
):
    model.train()
    train_acc = []
    test_acc = []
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        correct_train, total_train = 0, 0
        for i, (data, data_indices) in enumerate(train_loader):
            # data should have at least 2 samples, otherwise
            # it will fail at batch normalization layer
            if data.shape[0] < 2:
                continue
            inputs = data[:, :-1].float().to(device)
            labels = data[:, -1].long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # compute training accuracy per batch
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += len(labels)

        # compute training accuracy per epoch
        train_accuracy = correct_train / total_train
        model.eval()
        test_accuracy, _, _ = compute_accuracy(
            model, test_loader, criterion, device
        )
        model.train()
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        losses.append(epoch_loss)
        print(
            f"Model {model_id}, Epoch {epoch + 1}/{epochs}, "
            f"Loss: {epoch_loss:.4f}, "
            f"Train Acc: {train_accuracy * 100:.2f}%, "
            f"Test Acc: {test_accuracy * 100:.2f}%"
        )

    return train_acc, test_acc, losses


def compute_accuracy(model, test_loader, criterion, device='cpu'):
    correct = 0
    total = 0
    test_loss = 0
    label_counts = defaultdict(int)
    with torch.no_grad():
        for data, data_indices in test_loader:
            # data should have at least 2 samples, otherwise
            # it will fail at batch normalization layer
            if data.shape[0] < 2:
                continue
            inputs = data[:, :-1].float().to(device)  # Move inputs to device
            labels = data[:, -1].long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label in predicted.tolist():
                label_counts[label] += 1
    test_loss = test_loss / len(test_loader)
    test_accuracy = correct / total
    return test_accuracy, test_loss, label_counts


def eval_cnn(model_id, train_loader, test_loader, model, criterion):
    # Train set accuracy
    device = 'cpu'
    model = model.to(device)
    model.eval()
    train_accuracy, train_loss, label_counts = compute_accuracy(
        model, train_loader, criterion, device
    )
    print(
        f"Model {model_id} | Accuracy on train set: "
        f"{int(100 * train_accuracy)} %"
    )

    test_accuracy, test_loss, test_lbl_counts = compute_accuracy(
        model, test_loader, criterion, device
    )
    for lbl, count in test_lbl_counts.items():
        if lbl not in label_counts:
            label_counts[lbl] = count
        else:
            label_counts[lbl] += count
    print(
        f"Model {model_id} | Accuracy on test set: "
        f"{int(test_accuracy * 100)} %"
    )

    return train_accuracy, test_accuracy, train_loss, test_loss, label_counts


def save_cnn(
    model: nn.Module,
    metadata: Dict[str, any],
    name: str,
    round_number: int | None = None,
    output_dir: str | None = None,
):
    if output_dir is None:
        out_dir = Path(__file__).parent
    else:
        out_dir = Path(output_dir).absolute()
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    out_name = (
        f"{name}_round_{round_number}" if round_number is not None else name
    )
    model_path = Path(out_dir, f"{out_name}.torch")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    metadata_path_name = f"{out_name}_meta.npz"
    metadata_path = Path(out_dir, metadata_path_name)
    np.savez(metadata_path, **metadata, allow_pickle=True)
    print(f"Model metadata saved to {metadata_path}")
