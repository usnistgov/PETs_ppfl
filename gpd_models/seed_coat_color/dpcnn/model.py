from typing import Dict
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, x_len):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, 10)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(10, 8, 8)
        self.dropout2 = nn.Dropout(0.1)
        self.conv3 = nn.Conv1d(8, 6, 6)
        self.pool = nn.MaxPool1d(2)
        final_len = (x_len - 21) // 2
        self.ln1 = nn.LayerNorm([6, final_len])
        self.fc1 = nn.Linear(6 * final_len, 24)
        self.fc2 = nn.Linear(24, 16)
        self.fc3 = nn.Linear(16, 4)
        self.ln2 = nn.LayerNorm(4)

    def forward(self, x):
        # Convert to tensor if input is an ndarray
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(
                next(self.parameters()).device
            )
        x = x.float()
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.ln1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.ln2(x)
        x = F.softmax(x, dim=1)
        return x


def train_cnn(
    train_loader,
    test_loader,
    epochs,
    model,
    optimizer,
    criterion,
    delta,
    privacy_engine,
    device,
):
    model.train()
    train_acc = []
    test_acc = []
    eps_spent = []
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        correct_train, total_train = 0, 0
        for i, data in enumerate(train_loader):
            inputs = data[:, :-1].to(device)
            labels = data[:, -1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += len(labels)
        train_accuracy = (correct_train / total_train) * 100
        model.eval()
        test_accuracy, _, _ = compute_test_accuracy(
            model, test_loader, criterion, device
        )
        model.train()
        epsilon_spent, _ = privacy_engine.accountant.get_privacy_spent(
            delta=delta
        )
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        eps_spent.append(epsilon_spent)
        losses.append(epoch_loss)
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, "
            f"Train Acc: {train_accuracy:.2f}%, "
            f"Test Acc: {test_accuracy:.2f}%, "
            f"ε: {epsilon_spent:.2f}"
        )
    return train_acc, test_acc, eps_spent, losses


def compute_test_accuracy(model, test_loader, criterion, device):
    correct = 0
    total = 0
    test_loss = 0
    label_counts = defaultdict(int)
    with torch.no_grad():
        for data in test_loader:
            # data should have at least 2 samples, otherwise
            # it will fail at batch normalization layer
            if data.shape[0] < 2:
                continue
            inputs = data[:, :-1].to(device)  # Move inputs to device
            labels = data[:, -1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label in predicted.tolist():
                label_counts[label] += 1
    test_loss = test_loss / len(test_loader)
    test_accuracy = (correct / total) * 100
    return test_accuracy, test_loss, label_counts


def eval_cnn(train_loader, test_loader, model, criterion):
    # Train set accuracy
    correct = 0
    total = 0
    train_loss = 0
    label_counts = defaultdict(int)
    model = model.to('cpu')
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            # data should have at least 2 samples, otherwise
            # it will fail at batch normalization layer
            if data.shape[0] < 2:
                continue
            inputs = data[:, :-1]
            labels = data[:, -1]
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label in predicted.tolist():
                label_counts[label] += 1
    train_loss = train_loss / len(train_loader)
    train_accuracy = correct / total
    print(f"Accuracy on train set: {int(100 * correct / total)} %")

    test_accuracy, test_loss, test_lbl_counts = compute_test_accuracy(
        model, test_loader, criterion, 'cpu'
    )
    for lbl, count in test_lbl_counts.items():
        label_counts[lbl] += count
    print(f"Accuracy on test set: {int(test_accuracy)} %")

    return train_accuracy, test_accuracy, train_loss, test_loss, label_counts


def save_cnn(
    model: nn.Module,
    metadata: Dict[str, any],
    name: str,
    round_number: int | None = None,
):
    out_name = (
        f"{name}_round_{round_number}" if round_number is not None else name
    )
    parent_path = Path(__file__).parent
    model_path = Path(parent_path, f"{out_name}.torch")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    metadata_path_name = f"{out_name}_meta.npz"
    metadata_path = Path(parent_path, metadata_path_name)
    np.savez(metadata_path, **metadata)
    print(f"Model metadata saved to {metadata_path}")
