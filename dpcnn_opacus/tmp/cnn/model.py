from typing import Dict
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from report import Report

def unpack_batch(data, device=None):
    if isinstance(data, (tuple, list)):
        inputs, labels = data
    else:
        inputs = data[:, :-1]
        labels = data[:, -1]

    inputs = inputs.float()
    labels = labels.float()

    if device is not None:
        inputs = inputs.to(device)
        labels = labels.to(device)

    return inputs, labels

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
        flatten_len = self._calculate_flatten_size(features)
        self.layer_norm1 = nn.LayerNorm([8, flatten_len // 8])
        self.fc1 = nn.Linear(in_features=flatten_len, out_features=48)
        self.fc2 = nn.Linear(in_features=48, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.layer_norm2 = nn.LayerNorm(16)
        self.output = nn.Linear(in_features=16, out_features=1)

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
            return x.view(1, -1).size(1)

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
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool(x)
        x = self.layer_norm1(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.layer_norm2(x)
        x = self.output(x)
        return x


def train_cnn(
    model_id: int,
    train_loader,
    test_loader,
    epochs,
    model,
    optimizer,
    criterion,
    device='cpu'
):
    model.train()
    train_mse = []
    test_mse = []
    train_acc = []
    test_acc = []
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        total_mae, total_mse = 0, 0
        correct_train, pred_correct_train, total_train = 0, 0, 0
        for i, data in enumerate(train_loader):
            inputs, labels = unpack_batch(data, device)
            if inputs.shape[0] < 2:
                continue
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Use rounded-class accuracy
            pred_classes = torch.round(outputs)
            pred_correct = pred_classes == labels
            pred_correct_train += pred_correct.sum().item()
            total_train += len(labels)

            # Compute MAE and MSE per batch
            total_mae += torch.sum(torch.abs(outputs - labels.float())).item()
            total_mse += torch.sum((outputs - labels.float()) ** 2).item()
        
        # compute training accuracy per epoch
        train_accuracy = (pred_correct_train / total_train)
        mae = total_mae / total_train
        mse = total_mse / total_train
        rmse = mse**0.5

        model.eval()
        (
            epoch_test_mse,
            epoch_test_acc,
            _,
        ) = compute_test_accuracy(
            model, test_loader, criterion, device
        )
        model.train()
        train_mse.append(mse)
        train_acc.append(train_accuracy)
        test_mse.append(epoch_test_mse)
        test_acc.append(epoch_test_acc)
        losses.append(epoch_loss)
        print(
            f"Model {model_id} | "
            f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, "
            f"Train Acc: {train_accuracy:.2f}, "
            f"Test Acc: {epoch_test_acc:.2f}, "
            f"Train MSE: {mse:.4f}, Test MSE: {epoch_test_mse:.4f}, "
            f"MAE: {mae:.4f}, RMSE: {rmse:.4f}"
        )
    return train_mse, test_mse, train_acc, test_acc, losses


def compute_test_accuracy(model, test_loader, criterion, device):
    total_loss = 0
    total_mse = 0
    correct = 0
    total = 0
    pred_correct_test = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = unpack_batch(data, device)
            if inputs.shape[0] < 2:
                continue

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            total_mse += torch.sum((outputs - labels.float()) ** 2).item()
            
            pred_classes = torch.round(outputs)
            pred_correct = pred_classes == labels
            pred_correct_test += pred_correct.sum().item()
            total += labels.size(0)

            # Store predictions and labels for evaluation
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute testing accuracy per epoch
    accuracy = pred_correct_test / total
    average_loss = total_loss / len(test_loader)
    mse = total_mse / total
    return mse, accuracy, average_loss


def eval_cnn(model_id: int, train_loader, test_loader, model, criterion):
    def evaluate(loader):
        total_loss = 0
        total_mae = 0
        total_mse = 0
        correct = 0
        total = 0
        pred_correct_test = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for data in loader:
                inputs, labels = unpack_batch(data)
                if inputs.shape[0] < 2:
                    continue

                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                total_loss += loss.item()
                total_mae += torch.sum(torch.abs(outputs - labels)).item()
                total_mse += torch.sum((outputs - labels) ** 2).item()

                pred_classes = torch.round(outputs)
                pred_correct = pred_classes == labels
                pred_correct_test += pred_correct.sum().item()
                total += labels.size(0)

                # Store predictions and labels for evaluation
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute training accuracy per epoch
        accuracy = pred_correct_test / total
        average_loss = total_loss / len(loader)
        mae = total_mae / total
        mse = total_mse / total
        rmse = mse**0.5
        ss = sqrt(mean_squared_error(all_labels, all_preds))
        rr = r2_score(all_labels, all_preds)
        mm = np.mean(all_labels)
        error_mean = (ss / mm) * 100

        return (
            accuracy,
            average_loss,
            mae,
            mse,
            rmse,
            ss,
            rr,
            mm,
            error_mean,
            all_preds,
        )

    model.eval()
    model.to("cpu")
    (
        train_accuracy,
        train_loss,
        train_mae,
        train_mse,
        train_rmse,
        train_ss,
        train_rr,
        train_mm,
        train_error_mean,
        train_preds,
    ) = evaluate(train_loader)
    (
        test_accuracy,
        test_loss,
        test_mae,
        test_mse,
        test_rmse,
        test_ss,
        test_rr,
        test_mm,
        test_error_mean,
        test_preds,
    ) = evaluate(test_loader)

    print(f"\nModel {model_id} | Final Results:")
    print(
        f"Model {model_id} | Train: Accuracy: {train_accuracy:.2f}, "
        f"Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, "
        f"MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}"
    )
    print(f"Model {model_id} | R^2 Value is: {train_rr:.4f}")
    print(
        f"Model {model_id} | "
        f"RMSE for train set is: {train_ss:.4f} & mean is {train_mm:.4f}"
    )
    print(
        f"Model {model_id} | "
        f"This is {train_error_mean:.2f}% of the mean pheno data"
    )

    print(
        f"Model {model_id} | Test: Accuracy: {test_accuracy:.2f}, "
        f"Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, "
        f"MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}"
    )
    print(f"Model {model_id} | R^2 Value is: {test_rr:.4f}")
    print(
        f"Model {model_id} | "
        f"RMSE for test set is: {test_ss:.4f} & mean is {test_mm:.4f}"
    )
    print(
        f"Model {model_id} | "
        f"This is {test_error_mean:.2f}% of the mean pheno data"
    )

    return (
        train_accuracy,
        test_accuracy,
        train_loss,
        test_loss,
        train_mse,
        test_mse,
        train_preds,
        test_preds,
    )

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

    out_name = (
        f"{name}_round_{round_number}" if round_number is not None else name
    )
    model_path = Path(out_dir, f"{out_name}.torch")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    metadata_path_name = f"{out_name}_meta.npz"
    metadata_path = Path(out_dir, metadata_path_name)
    np.savez(metadata_path, **metadata, allow_pickle=True)
    report = Report(metadata)
    report.save_to_file(Path(out_dir, f"{out_name}.json"))
    print(f"Model metadata saved to {metadata_path}")
