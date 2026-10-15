## Understanding the Output <a name="output"></a>

### Printed Output

Currently printed output may look like:

```
Client 0 | Epoch 1/100 | ...
Client 3 | Epoch 1/100 | ...
Client 1 | Epoch 1/100 | ...
```

It means multiple client processes are training concurrently, and Ray prints logs as each process emits them. The order is based on scheduling/runtime progress, not client id order.

### File Output

Output files include `.npz` metadata files, `.json` human-readable reports, `.torch` CNN/DPCNN model files, and `.ubj` XGBoost model files. Within the `.npz` files, there is metadata on the model's global and client/round based performance. This is captured in the `.json` file format as well for human-readable purposes. The run also writes `input_parameters.json`, which records the resolved input parameters after schema/default processing. In the `.torch` and `.ubj` files, there are model weights that can be loaded for further inference with the trained model.

### Metric Definitions

<details>
<summary><strong> Client/Round Outputs </strong></summary>

| Metric | Description | Type | 
|---|---|---|
| `created on` | The datetime that the report was generated | string |
| `model id` | The id of the model being developed | integer |
| `round number` | The current round that the metrics are reporting on | integer |
| `partitions file` | The file used to partition the data for training | string |
| `problem type` | The task type used for the run | string |
| `class labels` | The configured class label order for classification runs, or `null` for regression | array or null |
| `train accuracy` | For classification, class prediction accuracy. For regression, fraction of predictions within `accuracy_tolerance`. | number |
| `test accuracy` | For classification, class prediction accuracy. For regression, fraction of predictions within `accuracy_tolerance`. | number |
| `train mean squared error` | Train MSE for this client and round. Classification reports set this to `0`. | number |
| `test mean squared error` | Test MSE for this client and round. Classification reports set this to `0`. | number |
| `train loss` | Train loss for this client and round | number |
| `test loss` | Test loss for this client and round | number |
| `train indices` | Indices used for training | integer array |
| `test indices` | Indices used for testing | integer array |
| `train accuracy per epoch` | Per-epoch training accuracy for CNN/DPCNN runs | number array |
| `test accuracy per epoch` | Per-epoch test accuracy for CNN/DPCNN runs | number array |
| `train mse per epoch` | Per-epoch training MSE for CNN/DPCNN regression runs. Classification reports use `0` values. | number array |
| `test mse per epoch` | Per-epoch test MSE for CNN/DPCNN regression runs. Classification reports use `0` values. | number array |
| `losses per epoch` | Loss values for each epoch in this client round. Present for CNN/DPCNN reports. | number array |
| `epsilon per epoch` | The epsilon values for each epoch in this client round. Present for DPCNN reports. | number array |
| `train predictions` | The predicted values for the training set | number array |
| `test predictions` | The predicted values for the test set | number array |
| `hyperparameters` | The hyperparameters the client used during its training process | object of the following properties: |
| `hyperparameters.learning rate` | Sets the model’s learning rate. This defines how much a model changes at each iteration | number | 
| `hyperparameters.weight decay` | Sets the model’s weight decay. This is a regularization method that penalizes high weights | number | 
| `hyperparameters.batch divisor` | The divisor to determine the number of batches (num_batches = dataset_size/batch_divisor) | integer |
| `hyperparameters.epochs` | The number of model training epochs. An epoch is one full pass through a training dataset | integer |
| `hyperparameters.seed` | The seed used to randomize training/testing | integer | 
| `hyperparameters.test fraction` | The fraction of the data set to set aside for testing | number |
| `hyperparameters.problem type` | The task type used for the run | string |
| `hyperparameters.class labels` | The configured class label order for classification runs, or `null` for regression | array or null |
| `hyperparameters.accuracy tolerance` | Regression accuracy tolerance. `null` for classification runs. | number or null |
| `hyperparameters.optimizer` | The optimizer is responsible for adjusting model parameters based on the value of the loss function | string |
| `hyperparameters.epsilon` | Determines the amount of privacy added to the data. | number |
| `hyperparameters.delta` | Measures the chance of a data breach. It defines the probability of the noise not adding sufficient privacy | number |
| `hyperparameters.max grad norm` | Clips the gradients to be under this maximum before adding noise | number |

</details>

<details>
<summary><strong> Global Outputs </strong></summary>

| Metric | Description | Type | 
|---|---|---|
| `created on` | The datetime that the report was generated | string |
| `loss per round` | Loss values across rounds | number array |
| `accuracy per round` | For classification, class prediction accuracy across rounds. For regression, tolerance-based accuracy across rounds for CNN/DPCNN and rounded-prediction accuracy for current XGBoost global evaluation. | number array |
| `mse per round` | MSE values across rounds. Classification reports use `0` values. | number array |

</details>

### Accuracy Calculations and Model Implications

Accuracy is calculated differently depending on `problem_type`.

For regression, model outputs are continuous numeric predictions. Accuracy is the fraction of predictions within the configured `accuracy_tolerance`:

```python
np.mean(np.abs(predictions - labels) <= accuracy_tolerance)
```

Regression reports also include MAE, MSE, RMSE, R2, the label mean, and RMSE as a percentage of the label mean.

For classification, labels are encoded from `class_labels`, and predictions are class indices. CNN/DPCNN classification uses the largest output logit as the predicted class. XGBoost classification uses `multi:softprob` probabilities when the default classification conversion is applied, then selects the class with the largest probability. Classification reports include accuracy, macro precision, macro recall, and macro F1. Regression-only metrics such as MSE are set to `0` in classification reports.

The older behavior of treating a regression output as a rounded class prediction is no longer the default task model. Use `problem_type: "classification"` for class labels and `problem_type: "regression"` for continuous numeric targets.
