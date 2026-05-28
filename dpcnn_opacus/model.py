from report import Report
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

####################################################################################################
#                                                                                                  #
#                                           Base Class                                             #
#                                                                                                  #
####################################################################################################

class BaseModel:
    def __init__(self, model_id=None, output_dir=None):
        self.model_id = model_id
        self.output_dir = output_dir
        self.model = None

    def output_name(self, name, round_number=None):
        return f"{name}_round_{round_number}" if round_number is not None else name

    def output_path(self, filename, output_dir=None):
        out_dir = Path(output_dir or self.output_dir or Path(__file__).parent).absolute()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / filename

    def save_metadata(self, metadata, name, round_number=None, output_dir=None):
        out_name = self.output_name(name, round_number)
        metadata_path = self.output_path(f"{out_name}_meta.npz", output_dir)
        np.savez(metadata_path, **metadata, allow_pickle=True)

    
    def save_report(self, metadata, name, round_number=None, output_dir=None):
        out_name = self.output_name(name, round_number)
        report = Report(metadata)
        report.save_to_file(self.output_path(f"{out_name}.json", output_dir))

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def get_parameters(self, *args, **kwargs):
        raise NotImplementedError

    def set_parameters(self, parameters):
        raise NotImplementedError

    def save_model(self, metadata, name, round_number=None, output_dir=None):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    def unpack_batch(self, data, device=None):
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

####################################################################################################
#                                                                                                  #
#                                         Callable Classes                                         #
#                                                                                                  #
####################################################################################################

class CNNModel(BaseModel):
    model_extension = ".torch"

    def build_model(self, num_features, output_dim=1):
        self.model = _CNNNet(num_features, output_dim=output_dim)
        return self.model
    
    def build_optimizer(self, optimizer_name, learning_rate, weight_decay):
        if self.model is None:
            raise RuntimeError("build_model must be called before build_optimizer")

        if optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        if optimizer_name == "adamax":
            return optim.Adamax(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

    def fit(self, train_loader, test_loader, epochs, optimizer, criterion, device='cpu'):
        self.model.train()
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
                inputs, labels = self.unpack_batch(data, device)
                if inputs.shape[0] < 2:
                    continue
                
                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
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

            self.model.eval()
            (
                epoch_test_mse,
                epoch_test_acc,
                _,
            ) = self.compute_test_metrics(test_loader, criterion, device)
            self.model.train()
            train_mse.append(mse)
            train_acc.append(train_accuracy)
            test_mse.append(epoch_test_mse)
            test_acc.append(epoch_test_acc)
            losses.append(epoch_loss)
            print(
                f"Model {self.model_id} | "
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, "
                f"Train Acc: {train_accuracy:.2f}, "
                f"Test Acc: {epoch_test_acc:.2f}, "
                f"Train MSE: {mse:.4f}, Test MSE: {epoch_test_mse:.4f}, "
                f"MAE: {mae:.4f}, RMSE: {rmse:.4f}"
            )
        return train_mse, test_mse, train_acc, test_acc, losses

    def evaluate(self, train_loader, test_loader, criterion):
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
                    inputs, labels = self.unpack_batch(data)
                    if inputs.shape[0] < 2:
                        continue

                    outputs = self.model(inputs).squeeze()
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

            return accuracy, average_loss, mae, mse, rmse, ss, rr, mm, error_mean, all_preds

        self.model.eval()
        self.model.to("cpu")
        (train_accuracy, train_loss, train_mae, train_mse, train_rmse,
         train_ss, train_rr, train_mm, train_error_mean, train_preds) = evaluate(train_loader)

        (test_accuracy, test_loss, test_mae, test_mse, test_rmse,
         test_ss, test_rr, test_mm,test_error_mean, test_preds) = evaluate(test_loader)

        print(f"\nModel {self.model_id} | Final Results:")
        print(f"Model {self.model_id} | Train: Accuracy: {train_accuracy:.2f}, "
              f"Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, "
              f"MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}")
        print(f"Model {self.model_id} | R^2 Value is: {train_rr:.4f}")
        print(f"Model {self.model_id} | "
              f"RMSE for train set is: {train_ss:.4f} & mean is {train_mm:.4f}")
        print(f"Model {self.model_id} | "
              f"This is {train_error_mean:.2f}% of the mean pheno data")

        print(f"Model {self.model_id} | Test: Accuracy: {test_accuracy:.2f}, "
              f"Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, "
              f"MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}")
        print(f"Model {self.model_id} | R^2 Value is: {test_rr:.4f}")
        print(f"Model {self.model_id} | "
              f"RMSE for test set is: {test_ss:.4f} & mean is {test_mm:.4f}")
        print(f"Model {self.model_id} | "
              f"This is {test_error_mean:.2f}% of the mean pheno data")

        return train_accuracy, test_accuracy, train_loss, test_loss, train_mse, test_mse, train_preds, test_preds

    def predict(self, data, device="cpu"):
        self.model.eval()

        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float32)

            data = data.float().to(device)

            if data.dim() == 2:
                data = data.unsqueeze(1)

            outputs = self.model(data).squeeze()

        return outputs.cpu().numpy()

    def compute_test_metrics(self, test_loader, criterion, device):
        total_loss = 0
        total_mse = 0
        correct = 0
        total = 0
        pred_correct_test = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = self.unpack_batch(data, device)
                if inputs.shape[0] < 2:
                    continue

                outputs = self.model(inputs).squeeze()
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
    
    def save_model(self, metadata, name, round_number=None, output_dir=None):
        out_name = self.output_name(name, round_number)
        model_path = self.output_path(f"{out_name}{self.model_extension}", output_dir)

        torch.save(self.model.state_dict(), model_path)
        self.save_metadata(metadata, name, round_number, output_dir)
        self.save_report(metadata, name, round_number, output_dir)

    def get_parameters(self, config=None):
        self.model.eval()
        return [
            val.detach().cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters):
        state_dict = {}

        for (key, ref_tensor), value in zip(self.model.state_dict().items(), parameters):
            state_dict[key] = torch.tensor(
                value,
                dtype=ref_tensor.dtype,
                device=ref_tensor.device,
            )

        self.model.load_state_dict(state_dict, strict=True)

    def load_model(self, path, num_features=None, output_dim=1, device="cpu"):
        if self.model is None:
            if num_features is None:
                raise ValueError("num_features is required when loading into an unbuilt model")
            self.build_model(num_features, output_dim=output_dim)

        state_dict = torch.load(path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        return self.model

class DPCNNModel(BaseModel):
    model_extension = ".torch"

    def build_model(self, num_features, output_dim=1):
        self.model = _CNNNet(num_features, output_dim=output_dim)
        return self.model
    
    def build_optimizer(self, optimizer_name, learning_rate, weight_decay):
        if self.model is None:
            raise RuntimeError("build_model must be called before build_optimizer")

        if optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay)

        if optimizer_name == "adamax":
            return optim.Adamax(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay)

    def attach_privacy_engine(self, optimizer, train_loader, opacus_params):
        privacy_engine = PrivacyEngine(
            accountant="rdp",
            secure_mode=opacus_params.get("secure_mode", False),
        )

        private_model, private_optimizer, private_train_loader = (
            privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=opacus_params.get("epochs", 1),
                target_epsilon=opacus_params.get("epsilon", 1.0),
                target_delta=opacus_params.get("delta", 1e-5),
                max_grad_norm=opacus_params.get("max_grad_norm", 1.0),
            )
        )

        privacy_engine.accountant.alphas = [1 + x / 10.0 for x in range(1000)]

        self.model = private_model
        return private_optimizer, private_train_loader, privacy_engine

    def fit(
        self,
        train_loader,
        test_loader,
        epochs,
        optimizer,
        criterion,
        delta,
        privacy_engine,
        tolerance=0.1,
        device="cpu",
    ):
        self.model.train()
        train_mse = []
        test_mse = []
        train_acc = []
        test_acc = []
        eps = []
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            total_mae, total_mse = 0, 0
            correct_train, pred_correct_train, total_train = 0, 0, 0
            for i, data in enumerate(train_loader):
                inputs, labels = self.unpack_batch(data, device)
                if inputs.shape[0] < 2:
                    continue
                
                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
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

            # Compute training accuracy per epoch
            train_accuracy = pred_correct_train / total_train
            mae = total_mae / total_train
            mse = total_mse / total_train
            rmse = mse**0.5

            # Compute test MSE per epoch
            self.model.eval()
            (
                epoch_test_mse,
                epoch_test_acc,
                _,
            ) = self.compute_test_metrics(test_loader, criterion, device)
            self.model.train()
            epsilon_spent, _ = privacy_engine.accountant.get_privacy_spent(
                delta=delta
            )
            train_mse.append(mse)
            train_acc.append(train_accuracy)
            test_mse.append(epoch_test_mse)
            test_acc.append(epoch_test_acc)
            losses.append(epoch_loss)
            eps.append(epsilon_spent)
            print(
                f"Model {self.model_id} | "
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, "
                f"Train Acc: {train_accuracy:.2f}, "
                f"Test Acc: {epoch_test_acc:.2f}, "
                f"Train MSE: {mse:.4f}, Test MSE: {epoch_test_mse:.4f}, "
                f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, "
                f"ε: {epsilon_spent:.2f}"
            )

        return train_mse, test_mse, train_acc, test_acc, losses, eps

    def evaluate(self, train_loader, test_loader, criterion):
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
                    inputs, labels = self.unpack_batch(data)
                    if inputs.shape[0] < 2:
                        continue

                    outputs = self.model(inputs).squeeze()
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

            return accuracy, average_loss, mae, mse, rmse, ss, rr, mm, error_mean, all_preds

        self.model.eval()
        self.model.to("cpu")
        (train_accuracy, train_loss, train_mae, train_mse, train_rmse,
         train_ss, train_rr, train_mm, train_error_mean, train_preds) = evaluate(train_loader)

        (test_accuracy, test_loss, test_mae, test_mse, test_rmse,
         test_ss, test_rr, test_mm,test_error_mean, test_preds) = evaluate(test_loader)

        print(f"\nModel {self.model_id} | Final Results:")
        print(f"Model {self.model_id} | Train: Accuracy: {train_accuracy:.2f}, "
              f"Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, "
              f"MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}")
        print(f"Model {self.model_id} | R^2 Value is: {train_rr:.4f}")
        print(f"Model {self.model_id} | "
              f"RMSE for train set is: {train_ss:.4f} & mean is {train_mm:.4f}")
        print(f"Model {self.model_id} | "
              f"This is {train_error_mean:.2f}% of the mean pheno data")

        print(f"Model {self.model_id} | Test: Accuracy: {test_accuracy:.2f}, "
              f"Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, "
              f"MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}")
        print(f"Model {self.model_id} | R^2 Value is: {test_rr:.4f}")
        print(f"Model {self.model_id} | "
              f"RMSE for test set is: {test_ss:.4f} & mean is {test_mm:.4f}")
        print(f"Model {self.model_id} | "
              f"This is {test_error_mean:.2f}% of the mean pheno data")

        return train_accuracy, test_accuracy, train_loss, test_loss, train_mse, test_mse, train_preds, test_preds

    def predict(self, data, device="cpu"):
        self.model.eval()

        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float32)

            data = data.float().to(device)

            if data.dim() == 2:
                data = data.unsqueeze(1)

            outputs = self.model(data).squeeze()

        return outputs.cpu().numpy()

    def compute_test_metrics(self, test_loader, criterion, device):
        total_loss = 0
        total_mse = 0
        correct = 0
        total = 0
        pred_correct_test = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = self.unpack_batch(data, device)
                if inputs.shape[0] < 2:
                    continue

                outputs = self.model(inputs).squeeze()
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
    
    def save_model(self, metadata, name, round_number=None, output_dir=None):
        out_name = self.output_name(name, round_number)
        model_path = self.output_path(f"{out_name}{self.model_extension}", output_dir)

        torch.save(self.model.state_dict(), model_path)
        self.save_metadata(metadata, name, round_number, output_dir)
        self.save_report(metadata, name, round_number, output_dir)

    def get_parameters(self, config=None):
        self.model.eval()
        return [
            val.detach().cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters):
        state_dict = {}

        for (key, ref_tensor), value in zip(self.model.state_dict().items(), parameters):
            state_dict[key] = torch.tensor(
                value,
                dtype=ref_tensor.dtype,
                device=ref_tensor.device,
            )

        self.model.load_state_dict(state_dict, strict=True)

    def load_model(self, path, num_features=None, output_dim=1, device="cpu"):
        if self.model is None:
            if num_features is None:
                raise ValueError("num_features is required when loading into an unbuilt model")
            self.build_model(num_features, output_dim=output_dim)

        state_dict = torch.load(path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        return self.model

class XGBoostModel(BaseModel):
    model_extension = ".ubj"

    def build_model(self, params):
        raise NotImplementedError

    def get_parameters(self, path):
        raise NotImplementedError

    def set_parameters(self, params):
        raise NotImplementedError

    def fit(self, train_data, test_data, num_local_round, train_method):
        raise NotImplementedError
    
    def evaluate(self, test_data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def save_model(self, metadata, name, round_number=None, output_dir=None):
        out_name = self.output_name(name, round_number)
        model_path = self.output_path(f"{out_name}{self.model_extension}", output_dir)

        self.model.save_model(model_path)
        self.save_metadata(metadata, name, round_number, output_dir)
        self.save_report(metadata, name, round_number, output_dir)

    def load_model(self, path):
        raise NotImplementedError

    def save_raw_parameters(self, path):
        raise NotImplementedError

####################################################################################################
#                                                                                                  #
#                                       Helper Model Class                                         #
#                                                                                                  #
####################################################################################################

class _CNNNet(nn.Module):
    def __init__(self, features: int, output_dim: int = 1):
        super().__init__()
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
        self.output = nn.Linear(in_features=16, out_features=output_dim)

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