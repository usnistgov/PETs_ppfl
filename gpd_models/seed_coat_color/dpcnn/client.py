from collections import OrderedDict
from pathlib import Path
import json
from datetime import datetime

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine

from dataset import (
    load_pickle_data,
    load_custom_partitions,
    load_random_partitions,
)
from model import Net, eval_cnn, save_cnn, train_cnn
from utils import client_args_parser, get_device


# warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = get_device()
print('DEVICE: ', DEVICE)
# Parse arguments for client parameters
args = client_args_parser()
print(f"client args: {args}")
partitioner_type = args.partitioner_type
client_id = args.client_id
partition_id = args.partition_id
num_partitions = args.num_partitions
batch_divisor = args.batch_divisor
learning_rate = args.learning_rate
weight_decay = args.weight_decay
seed = args.seed
test_fraction = args.test_fraction
epochs = args.epochs
data_partitions_file = args.data_partitions_file
n_models = args.n_models
optimizer_name = args.optimizer

# Privacy parameter
epsilon = args.epsilon  # Target privacy budget (epsilon)
delta = args.delta  # Target delta
max_grad_norm = args.max_grad_norm  # param to clip the gradients
opacus_secure_mode = args.opacus_secure_mode  # Use Opacus secure mode

ohe, tt_vcf, tt_pheno = load_pickle_data()
combined_dataset = np.concatenate((tt_vcf, tt_pheno), axis=1)

# Check if data partitions file is provided and exists
data_partitions = None
if data_partitions_file and Path(data_partitions_file).exists():
    data_partitions = np.load(data_partitions_file)
    print(f"Loaded data partitions from {data_partitions_file}")
else:
    print(
        f"Data partitions file not found at {data_partitions_file}. "
        f"Using a random partition for client {client_id}"
    )

batch_size = max(1, tt_vcf.shape[0] // batch_divisor)

if data_partitions is not None:
    # if data partitions available, train a model for each data partition
    train_loader, test_loader, train_indices, test_indices = (
        load_custom_partitions(
            combined_dataset,
            client_id,
            data_partitions,
            test_fraction,
            batch_size,
            seed,
        )
    )
else:
    # Partition data into n_models randomly to train N client models
    train_loader, test_loader, train_indices, test_indices = (
        load_random_partitions(
            combined_dataset,
            partition_id,
            partitioner_type,
            num_partitions,
            test_fraction,
            batch_size,
            seed,
        )
    )

model = Net(tt_vcf.shape[1])
criterion = None
optimizer = None
privacy_engine = None


def init_model():
    global model, criterion, optimizer, privacy_engine, train_loader
    model = Net(tt_vcf.shape[1])
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "adamax":
        optimizer = optim.Adamax(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

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


def save_client(
    federated_round,
    train_acc,
    test_acc,
    train_loss,
    test_loss,
    train_acc_per_epoch,
    test_acc_per_epoch,
    epsilon_spent_per_epoch,
    losses_per_epoch,
    predicted_labels,
):
    partitions_path = (
        Path(data_partitions_file).name if data_partitions else 'none'
    )
    metadata = {
        'created on': str(datetime.now()),
        'model id': int(client_id),
        'partitions file': partitions_path,
        'train accuracy': float(train_acc),
        'test accuracy': float(test_acc),
        'train loss': float(train_loss),
        'test loss': float(test_loss),
        'train indices': train_indices,
        'test indices': test_indices,
        "train accuracy per epoch": np.array(train_acc_per_epoch),
        "test accuracy per epoch": np.array(test_acc_per_epoch),
        "epsilon per epoch": np.array(epsilon_spent_per_epoch),
        "losses per epoch": np.array(losses_per_epoch),
        "predicted labels": json.dumps(predicted_labels),
        'hyperparameters': {
            'learning rate': float(learning_rate),
            'weight decay': float(weight_decay),
            'batch divisor': int(batch_divisor),
            'epochs': int(epochs),
            'seed': int(seed),
            'test fraction': float(test_fraction),
            "optimizer": optimizer_name,
            "epsilon": float(epsilon),
            "delta": float(delta),
            "max grad norm": float(max_grad_norm),
        },
    }

    metadata['hyperparameters'] = json.dumps(metadata['hyperparameters'])
    save_cnn(model, metadata, f'fldpcnn_{client_id}', federated_round)


def train_partition():
    # train model
    model.to(DEVICE)
    train_acc, test_acc, eps_spent, losses = train_cnn(
        train_loader,
        test_loader,
        epochs,
        model,
        optimizer,
        criterion,
        delta,
        privacy_engine,
        DEVICE,
    )
    return train_acc, test_acc, eps_spent, losses


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        model.eval()
        return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).float().to(DEVICE)
                                  for k, v in params_dict})
        model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        init_model()
        self.set_parameters(parameters)
        self.current_round = config.get('server_round', 1) - 1
        (
            self.train_acc_per_epoch,
            self.test_acc_per_epoch,
            self.eps_spent,
            self.losses,
        ) = train_partition()
        return (
            self.get_parameters(config=config),
            len(train_loader.dataset),
            {},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        train_acc, test_acc, train_loss, test_loss, predicted_labels = (
            eval_cnn(train_loader, test_loader, model, criterion)
        )
        save_client(
            self.current_round,
            train_acc,
            test_acc,
            train_loss,
            test_loss,
            self.train_acc_per_epoch,
            self.test_acc_per_epoch,
            self.eps_spent,
            self.losses,
            predicted_labels,
        )
        # convert loss to float to avoid Flower-Numpy type error
        return (
            float(test_loss),
            len(test_loader.dataset),
            {"accuracy": test_acc},
        )


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient().to_client(),
)
