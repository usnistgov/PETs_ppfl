from typing import Dict
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine


class Net(nn.Module):
    def __init__(self, features: int):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 24, kernel_size=14, padding=1)
        self.dropout1 = nn.Dropout1d(0.2)
        self.conv2 = nn.Conv1d(24, 16, kernel_size=10, padding=1)
        self.dropout2 = nn.Dropout1d(0.1)
        self.conv3 = nn.Conv1d(16, 8, kernel_size=8, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        flatten_size = self._calculate_flatten_size(features)
        self.layer_norm1 = nn.LayerNorm([8, flatten_size//8])
        self.fc1 = nn.Linear(
            in_features=flatten_size, out_features=48
        )
        self.fc2 = nn.Linear(in_features=48, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.layer_norm2 = nn.LayerNorm(16)
        self.output = nn.Linear(in_features=16, out_features=2)

    def _calculate_flatten_size(self, features: int):
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.zeros(1, 1, features).to(device)
            x = self.conv1(x)
            x = F.relu(x)
            x = self.dropout1(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.conv3(x)
            x = F.relu(x)
            x = self.pool(x)
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
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.layer_norm1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.layer_norm2(x)
        x = self.output(x)
        x = F.softmax(x, dim=1)
        return x


def train_cnn(
    train_loader,
    test_loader,
    epochs,
    model,
    optimizer,
    criterion,
    epsilon,
    delta,
    max_grad_norm,
    opacus_secure_mode,
    device='cpu',
):
    # Initialize PrivacyEngine
    privacy_engine = PrivacyEngine(
        accountant='rdp', secure_mode=opacus_secure_mode
    )
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=max_grad_norm,
    )
    privacy_engine.accountant.alphas = [1 + x / 10.0 for x in range(1000)]

    print(f"Training with DP: epsilon = {epsilon}")
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
