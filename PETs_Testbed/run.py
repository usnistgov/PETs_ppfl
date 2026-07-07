# This Software (PETs Testbed) is being made available as a public service by the
# National Institute of Standards and Technology (NIST), an Agency of the United
# States Department of Commerce. This software was developed in part by employees of
# NIST and in part by NIST contractors. Copyright in portions of this software that
# were developed by NIST contractors has been licensed or assigned to NIST. Pursuant
# to Title 17 United States Code Section 105, works of NIST employees are not
# subject to copyright protection in the United States. However, NIST may hold
# international copyright in software created by its employees and domestic
# copyright (or licensing rights) in portions of software that were assigned or
# licensed to NIST. To the extent that NIST holds copyright in this software, it is
# being made available under the Creative Commons Attribution 4.0 International
# license (CC BY 4.0). The disclaimers of the CC BY 4.0 license apply to all parts
# of the software developed or licensed by NIST.
#
# ACCESS THE FULL CC BY 4.0 LICENSE HERE:
# https://creativecommons.org/licenses/by/4.0/legalcode

import os

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

from utils import get_device, ConfigPipeline, configure_warning_logging
from report import Report
from dataset import ensure_npy_feature_label_files

# Parse arguments for flower server and client
pipeline = ConfigPipeline()
args = pipeline.parse()
print("Parameter values:\n")
args.print()

if args.check_only:
    print("Parameters validated. Ending script")
    exit()
    
#base params
model_type = args.model_type
num_cpus = args.num_cpus
num_gpus = args.num_gpus
out_dir = args.output_dir
data_dir = args.data_dir
data_partitions_file = args.data_partitions_file
partitioner_type = args.partitioner_type
num_partitions = args.num_partitions
seed = args.seed
problem_type = args.problem_type
class_labels = args.class_labels if problem_type == "classification" else None
num_classes = len(class_labels) if problem_type == "classification" else 1
accuracy_tolerance = args.accuracy_tolerance if problem_type == "regression" else None

#common params
partition_id = args.model_params["partition_id"]
client_id = args.model_params["client_id"]
epochs = args.model_params["epochs"]
batch_divisor = args.model_params["batch_divisor"]
test_fraction = args.model_params["test_fraction"]

learning_rate = None; weight_decay = None; optimizer_name = None
epsilon = None; delta = None; max_grad_norm = None; opacus_secure_mode = None
train_method = None; centralised_eval = None; scaled_lr = None; xgboost_params=None


#Federated params
num_rounds = args.federated["num_rounds"]
num_clients = args.federated["num_clients"]

if model_type == "dpcnn":
    learning_rate = args.model_params["learning_rate"]
    weight_decay = args.model_params["weight_decay"]
    optimizer_name = args.model_params["optimizer"]

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

    xgboost_params = {
        key: args.model_params[key]
        for key in [
            "objective",
            "eta",
            "max_depth",
            "eval_metric",
            "nthread",
            "num_parallel_tree",
            "subsample",
            "tree_method",
            "colsample_bylevel",
            "colsample_bytree",
            "gamma",
            "max_delta_step",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
            "scale_pos_weight",
        ]
    }
    if problem_type == "classification":
        xgboost_params["num_class"] = num_classes
        if xgboost_params.get("objective") == "reg:squarederror":
            xgboost_params["objective"] = "multi:softprob"
        if xgboost_params.get("eval_metric") == "rmse":
            xgboost_params["eval_metric"] = "mlogloss"
else:
    print("error - unhandled model type")
    exit(1)

out_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
data_dir = args.data_dir
ensure_npy_feature_label_files(data_dir)

# Create output directory
if out_dir is None:
    out_dir = Path(__file__).parent
else:
    out_dir = Path(out_dir).absolute()
if not out_dir.exists():
    out_dir.mkdir(parents=True)

if not args.print_warning_logs:
    configure_warning_logging(out_dir)

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

from client import FlowerClient
from server import create_strategy
client_params = {
    # partitioner_type ->  uniform, linear, square, exponential
    "model_type": model_type,
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
    'train_method': train_method,
    'centralised_eval': centralised_eval,
    'scaled_lr': scaled_lr,
    'xgboost_params': xgboost_params,
    'print_warning_logs': args.print_warning_logs,
    'problem_type': problem_type,
    'class_labels': class_labels,
    'num_classes': num_classes,
}
def client_fn(context: Context):
    """Returns a FlowerClient"""
    client_id = context.node_config["partition-id"]
    return FlowerClient(context, client_id, client_params).to_client()


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""
    strategy = create_strategy(
        {
            "model_type": model_type,
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'accuracy_tolerance': accuracy_tolerance,
            'output_dir': out_dir,
            'data_dir': data_dir,
            'batch_divisor': batch_divisor,
            'train_method': train_method,
            'centralised_eval': centralised_eval,
            'scaled_lr': scaled_lr,
            'xgboost_params': xgboost_params,
            'problem_type': problem_type,
            'class_labels': class_labels,
            'num_classes': num_classes,
        }
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Construct the ClientApp passing the client generation function
client_app = ClientApp(client_fn=client_fn)
server_app = ServerApp(server_fn=server_fn)

DEVICE = get_device()

if DEVICE.type != 'cpu':
    backend_config = {"client_resources": {"num_cpus": args.num_cpus, "num_gpus": args.num_gpus}}
else:
    backend_config = {"client_resources": {"num_cpus": args.num_cpus, "num_gpus": 0}} 

print('BACKEND CONFIG', backend_config)
run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=num_partitions,
    backend_config=backend_config,
)
