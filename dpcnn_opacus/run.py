import importlib.util
import os
import sys

# Stop de-duplicating logs in Ray
os.environ["RAY_DEDUP_LOGS"] = "0"
import re
from typing import Dict
from pathlib import Path
import numpy as np
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from datetime import datetime

from utils import get_device, ConfigPipeline
from report import Report

# Parse arguments for flower server and client
'''
ORIGINAL CODE:

    args = flower_args_parser()

RECOMMENDED CODE:
'''
pipeline = ConfigPipeline()
args = pipeline.parse()
print("Parameter values:\n")
args.print()

print("Parameter values:\n")
args.print()

if args.check_only:
    print("Parameters validated. Ending script")
    exit()

if args.model_type == "xgboost":
    print("That functionality has not been implemented yet. Terminating process.")
    exit(0)
    
'''
INTENDED ACTION: Modify

JUSTIFICAITON: Testing new parameterization methods
'''

#base params
model_type = args.model_type
num_cpus = args.num_cpus
num_gpus = args.num_gpus
out_dir = args.output_dir
data_dir = args.data_dir

#common params
data_partitions_file = args.model_params["data_partitions_file"]
partitioner_type = args.model_params["partitioner_type"]
num_partitions = args.model_params["num_partitions"]
partition_id = args.model_params["partition_id"]
client_id = args.model_params["client_id"]
seed = args.model_params["seed"]
epochs = args.model_params["epochs"]
batch_divisor = args.model_params["batch_divisor"]
test_fraction = args.model_params["test_fraction"]

learning_rate = None; weight_decay = None; optimizer_name = None
accuracy_tolerance = None; epsilon = None; delta = None; max_grad_norm = None; opacus_secure_mode = None
train_method = None; centralised_eval = None; scaled_lr = None


#Federated params
num_rounds = args.federated["num_rounds"]
min_fit_clients = args.federated["min_fit_clients"]
min_evaluate_clients = args.federated["min_evaluate_clients"]
min_available_clients = args.federated["min_available_clients"]

if model_type == "dpcnn":
    learning_rate = args.model_params["learning_rate"]
    weight_decay = args.model_params["weight_decay"]
    optimizer_name = args.model_params["optimizer"]
    accuracy_tolerance = args.model_params["accuracy_tolerance"]

    # Privacy arguments
    epsilon = args.dp["epsilon"]  # Target privacy budget (epsilon)
    delta = args.dp["delta"]  # Target delta
    max_grad_norm = args.dp["max_grad_norm"]  # param to clip the gradients
    opacus_secure_mode = args.dp["opacus_secure_mode"]  # Use Opacus secure mode
elif model_type == "cnn":
    learning_rate = args.model_params["learning_rate"]
    weight_decay = args.model_params["weight_decay"]
    optimizer_name = args.model_params["optimizer"]
elif model_type == "xgboost":
    train_method = args.model_params["train_method"]
    centralised_eval = args.model_params["centralised_eval"]
    scaled_lr = args.model_params["scaled_lr"]
else:
    print("error - unhandled model type")
    exit(1)

'''
# server arguments
num_rounds = args.federated["num_rounds"]
min_fit_clients = args.federated["min_fit_clients"]
min_evaluate_clients = args.federated["min_evaluate_clients"]
min_available_clients = args.federated["min_available_clients"]

# client arguments
partitioner_type = args.model_params["partitioner_type"]
num_partitions = args.model_params["num_partitions"]
batch_divisor = args.model_params["batch_divisor"]
learning_rate = args.model_params["learning_rate"]
weight_decay = args.model_params["weight_decay"]
seed = args.model_params["seed"]
test_fraction = args.model_params["test_fraction"]
epochs = args.model_params["epochs"]
accuracy_tolerance = args.model_params["accuracy_tolerance"]
data_partitions_file = args.model_params["data_partitions_file"]
out_dir = args.output_dir

RECOMMENDED CODE:
'''
out_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
'''
INTENDED ACTION: Modify
JUSTIFICATION: Creates a datetime folder on a run on top of the previously assigned directory. Provides additional separation for runs by default
'''
data_dir = args.data_dir
optimizer_name = args.model_params["optimizer"]

# Create output directory
if out_dir is None:
    out_dir = Path(__file__).parent
else:
    out_dir = Path(out_dir).absolute()
if not out_dir.exists():
    out_dir.mkdir(parents=True)

# Save these parameters into a json report
arg_dictionary = vars(args)
# remove unneeded "_print_schema" field from input parameters report
arg_dictionary.pop("_print_schema")
parameter_report = Report(arg_dictionary)
parameter_path = Path(out_dir, f"input_parameters.json")
parameter_report.save_to_file(parameter_path)
print(f"Input parameters saved to {parameter_path}")

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

#variable imports
file_path = "tmp/" + args.model_type

print(file_path)

sys.path.append(file_path)

from client import FlowerClient
from server import create_strategy

client_params = {
    # partitioner_type ->  uniform, linear, square, exponential
    'partitions_type': partitioner_type,
    'num_partitions': num_partitions,
    'batch_divisor': batch_divisor,
    'partition_id': partition_id,
    'learning_rate': learning_rate,
    'weight_decay': weight_decay,
    'epochs': epochs,
    'seed': seed,
    'test_fraction': test_fraction,
    'accuracy_tolerance': accuracy_tolerance,
    'data_partitions_file': data_partitions_file,
    'output_dir': out_dir,
    'data_dir': data_dir,
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
            'data_dir': data_dir
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

if DEVICE.type != 'cpu':
    backend_config = {"client_resources": {"num_cpus": args.num_cpus, "num_gpus": args.num_gpus}}
else:
    backend_config = {"client_resources": {"num_cpus": args.num_cpus, "num_gpus": 0}} 

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
