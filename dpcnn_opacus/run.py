import os

# Stop de-duplicating logs in Ray
os.environ["RAY_DEDUP_LOGS"] = "0"
import re
from pathlib import Path
import numpy as np
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context

from client import FlowerClient
from server import create_strategy
from utils import flower_args_parser, json_args_parser, get_device


# Parse arguments for flower server and client
'''
ORIGINAL CODE:

    args = flower_args_parser()

RECOMMENDED CODE:
'''

args = json_args_parser()
if args.check_only:
    print("Parameter values:")
    for elem in vars(args):
        print(f"\t{elem}={getattr(args, elem)}")
    print("Parameters validated. Ending script")
    exit()

'''
INTENDED ACTION: Modify

JUSTIFICAITON: Testing new parameterization methods
'''

print(f"flower args: {args}")
# server arguments
num_rounds = args.num_rounds
min_fit_clients = args.min_fit_clients
min_evaluate_clients = args.min_evaluate_clients
min_available_clients = args.min_available_clients

# client arguments
partitioner_type = args.partitioner_type
num_partitions = args.num_partitions
batch_divisor = args.batch_divisor
learning_rate = args.learning_rate
weight_decay = args.weight_decay
seed = args.seed
test_fraction = args.test_frac
epochs = args.epochs
accuracy_tolerance = args.accuracy_tolerance
data_partitions_file = args.data_partitions_file
out_dir = args.output_dir
optimizer_name = args.optimizer

# Privacy arguments
epsilon = args.epsilon  # Target privacy budget (epsilon)
delta = args.delta  # Target delta
max_grad_norm = args.max_grad_norm  # param to clip the gradients
opacus_secure_mode = args.opacus_secure_mode  # Use Opacus secure mode

# Get number of partitions from data_partitions_file
# if it exists and is not None
data_partitions = None
if data_partitions_file and Path(data_partitions_file).exists():
    data_partitions = np.load(data_partitions_file)
    data_partition_ids = sorted(
        [k for k in data_partitions.keys() if re.match(r'client_\d+', k)]
    )
    num_partitions = len(data_partition_ids)

    '''
    ORIGINAL CODE:

min_fit_clients = num_partitions
min_evaluate_clients = num_partitions
min_available_clients = num_partitions

    RECOMMENDED CODE:
    '''

    min_fit_clients = num_partitions
    min_evaluate_clients = num_partitions
    min_available_clients = num_partitions

    '''
    INTENDED ACTION: Modify

    JUSTIFICAITON: Original code was outside intended if statement
    '''


client_params = {
    # partitioner_type ->  uniform, linear, square, exponential
    'partitioner_type': partitioner_type,
    'num_partitions': num_partitions,
    'batch_division': batch_divisor,
    'learning_rate': learning_rate,
    'weight_decay': weight_decay,
    'epochs': epochs,
    'seed': seed,
    'test_fraction': test_fraction,
    'accuracy_tolerance': accuracy_tolerance,
    'data_partitions_file': data_partitions_file,
    'output_dir': out_dir,
    'optimizer_name': optimizer_name,
    'epsilon': epsilon,
    'delta': delta,
    'max_grad_norm': max_grad_norm,
    'opacus_secure_mode': opacus_secure_mode,
}

def client_fn(context: Context):
    """Returns a FlowerClient"""
    client_id = context.node_config["partition-id"]
    return FlowerClient(context, client_id, client_params).to_client()


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""
    strategy = create_strategy(
        {
            'min_fit_clients': min_fit_clients,
            'min_evaluate_clients': min_evaluate_clients,
            'min_available_clients': min_available_clients,
            'num_rounds': num_rounds,
            'accuracy_tolerance': accuracy_tolerance,
            'output_dir': out_dir,
        }
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Construct the ClientApp passing the client generation function
client_app = ClientApp(client_fn=client_fn)
server_app = ServerApp(server_fn=server_fn)

DEVICE = get_device()

'''
ORIGINAL CODE:

backend_config = None

if DEVICE.type != 'cpu':
    backend_config = {"client_resources": {"num_cpus": 2, "num_gpus": 1}}

RECOMMENDED CODE:
'''

backend_config = {"client_resources": {"num_cpus": 12, "num_gpus": 0}}

if DEVICE.type != 'cpu':
    backend_config = {"client_resources": {"num_cpus": 8, "num_gpus": 1}}

'''
INTENDED ACTION: Modify

JUSTIFICAITON: Increasing CPUs to speed up performance
'''

print('BACKEND CONFIG', backend_config)
run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=num_partitions,
    backend_config=backend_config,
)
