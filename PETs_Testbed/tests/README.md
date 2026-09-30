# PETs Testbed test suite

This directory holds the automated tests for the PETs privacy-preserving federated
learning (FL) testbed. The tests serve three purposes: they catch regressions in the
command-line and configuration front end, they exercise the individual helper functions
that make up the data, client, server, and reporting layers, and they pin down the
differential-privacy (DP) accounting behavior that the DP-CNN training path relies on.

If you are new to the project, start with the table below, then open the script you
are interested in. Every script begins with a module docstring that describes each test
it contains.

## Running the tests

Run everything from the `PETs_Testbed` directory with the virtual environment active:

```bash
cd PETs_Testbed
python3.12 -m pytest -q
```

`pytest.ini` in `PETs_Testbed` sets `testpaths = tests` and deselects the `perf` marker
by default, so a plain run executes the regression and unit tests only. Two opt-in
switches exist:

| Switch | Effect |
| --- | --- |
| `--run-e2e` | Also run regression case `t46`, a real end-to-end training run. Skipped by default because it takes minutes. |
| `-m perf` | Run only the performance benchmarks in `test_dataset_perf.py`. Requires `pytest-benchmark`. |

To run a single file or test:

```bash
python3.12 -m pytest -q tests/test_dataset.py
python3.12 -m pytest -q "tests/test_inputs.py::test_run_py_regressions[t12a]"
```

The `tests/__init__.py` file makes this directory a package, which lets pytest put
`PETs_Testbed` on the import path so tests can `import dataset`, `import server`, and so
on without any path manipulation.

## Files at a glance

| File | Kind | Tests | Module under test | What it checks |
| --- | --- | --- | --- | --- |
| `conftest.py` | Fixtures | n/a | n/a | Shared fixtures, the `--run-e2e` option, and a macOS OpenMP workaround. |
| `test_inputs.py` | Regression | 2 functions, 102 cases | `run.py`, `utils.py` | Config-file and CLI parsing, schema validation, error messages, and printed output order. |
| `test_utils_config.py` | Unit | 27 | `utils.py` | The schema-driven configuration engine: coercion, defaults, layering, pruning, validation. |
| `test_dataset.py` | Unit | 29 | `dataset.py` | Label encoding, memory-mapped indexed datasets, train/test splitting, `.dat` to `.npy` conversion. |
| `test_dataset_perf.py` | Performance | 2 | `dataset.py` | Throughput of random row access and bulk label gathers. Deselected by default. |
| `test_client_helpers.py` | Unit | 3 | `client.py` | Empty evaluation results and loader-to-XGBoost `DMatrix` conversion. |
| `test_server_helpers.py` | Unit | 7 | `server.py` | Weighted metric aggregation, model-class dispatch, per-round history bookkeeping. |
| `test_model_metrics.py` | Unit | 5 | `model.py` | Regression and classification metric computation shared by the Torch and XGBoost models. |
| `test_reports.py` | Unit | 5 | `report.py` | JSON serialization of run reports, including NumPy values and known gaps. |
| `test_dp_privacy.py` | Privacy | 8 | `model.py`, Opacus | The Renyi DP (RDP) accountant contract: no degenerate orders, no custom alpha grid, bounded slack. |

## Script descriptions

### `conftest.py`

Pytest loads this file before any test module. It does three things:

- **Pins OpenMP to one thread.** On macOS, `torch` and `xgboost` each ship their own
  OpenMP runtime. Importing both can crash the process the first time XGBoost enters a
  parallel region. Setting `OMP_NUM_THREADS=1` before either import avoids that. This
  affects only the test process.
- **Registers the `--run-e2e` option** used by `test_inputs.py` to enable the
  end-to-end smoke test.
- **Provides three fixtures** that build small synthetic datasets in a temporary
  directory. `synthetic_split` returns two in-memory feature/label array pairs of
  different lengths so tests can exercise the boundary between backing arrays.
  `npy_data_dir` writes the six `*_vcf.npy` and `*_pheno.npy` files the loaders expect.
  `dat_data_dir` writes pickled `.dat` arrays plus one non-array pickle for the
  conversion tests.

No test data in this suite is real. Everything is generated with a fixed random seed.

### `test_inputs.py`

The regression suite for the program entry point. Each case writes a configuration file
to a temporary directory, launches `run.py --check_only` in a subprocess with optional
CLI overrides, and asserts on the exit code and on regular expressions that must or
must not appear in stdout. Because it only runs the validation stage, the whole suite
finishes in well under a minute.

Cases are `Case` dataclass instances collected in the `CASES` list and grouped as:

- **A. Config and CLI precedence.** Config-only load, single and multiple CLI overrides,
  repeated flags, unknown arguments and keys, missing or malformed config files, type
  coercion, and path quoting.
- **B. Per-parameter validation.** For most schema fields, one case at a valid boundary
  and one just past it: round and client counts, partition settings, seed, epochs,
  batch size, test fraction, learning rate, weight decay, epsilon, delta, gradient norm,
  accuracy tolerance, and the enumerated string fields.
- **C. Cross-parameter interactions.** Class label constraints, partition ID versus
  partition count, partition file versus partitioner type, DP gating.
- **D. Error-message regression.** Messages name the parameter, the expected form and
  the received value, exit codes are non-zero on failure, and multiple errors are
  aggregated into one report.
- **E. End-to-end scenarios.** Representative FL configs with and without a partitions
  file, maximal valid bounds, and the `t46` smoke run that trains for real when
  `--run-e2e` is given.
- **F and G. Schema and CLI behavior.** Model-type selection between `dpcnn`, `cnn`, and
  `xgboost`, branch-specific parameters that apply to one model being ignored for
  another, and the rule that bad CLI values are reported by schema validation rather
  than by `argparse`.

A second test, `test_check_only_print_order_follows_effective_schema_top_level`, checks
that the parameters `run.py` echoes back are printed in the order the effective schema
defines them.

The helper functions `_base_with`, `_cnn_base_with`, and `_xgb_base_with` build a valid
baseline config for each model type with selected overrides. See
`docs/unit_tests.md` for a walkthrough of adding a new case.

### `test_utils_config.py`

Direct unit tests for the configuration engine in `utils.py`, which the regression suite
above only reaches through a subprocess. Sections cover:

- `_coerce_cli_value` and `_strtobool`: turning CLI strings into typed values, including
  the truthy and falsy vocabulary inherited from the removed `distutils.util.strtobool`,
  and falling back to the raw string when coercion fails so the schema reports the error.
- `apply_defaults`: filling scalar and nested defaults, respecting supplied values, and
  choosing the correct `oneOf` branch from the `model_type` discriminator.
- `_to_layered_config`: nesting flat keys into the `federated`, `dp`, and `model_params`
  groups defined by the schema.
- `_prune_inactive_model_params`: dropping parameters that belong to a different model
  type or problem type while keeping genuinely unknown keys so they can be reported.
- `_sync_enabled_flags`: inferring `federated_enabled` and `dp_enabled` from whether the
  group is populated, unless the user set the flag explicitly.
- `_get_validation_errors` and `UnknownParameterError`: normalized messages for missing
  required fields, unknown properties, type mismatches, and did-you-mean suggestions.
- `validate_file_path`, `validate_dir_path`, and `validate_data_size`: filesystem checks,
  including that a batch size cannot exceed the dataset size found in `.npy` or `.dat`
  files.

### `test_dataset.py`

Unit tests for the data layer in `dataset.py`, which loads genotype features and
phenotype labels from memory-mapped `.npy` files without copying them into RAM. See
`docs/memory_mapped_data_loading.md` for the design. Sections cover:

- `normalize_label` and `build_label_to_index`: unwrapping NumPy scalars and mapping
  class labels to contiguous integer positions.
- `IndexedArrayDataset`: a PyTorch dataset that presents several backing arrays as one
  logical table. Tests check length, index resolution across the array boundary, index
  ordering, reshaping of `(n, 1)` labels, classification label encoding, and the errors
  raised for unknown labels or mismatched array lists.
- `get_split_labels`: gathering labels for a set of global indices, including empty
  index lists of several dtypes and out-of-range indices.
- `train_test_indices_split` and `train_test_split_backed_indices`: the split is
  disjoint and complete, deterministic under a seed, keeps singleton classes in the
  training set, bins regression targets before stratifying, and returns global indices.
- `find_single_npy`, `convert_dat_to_npy`, `load_npy_feature_label_data`, and
  `resolve_data_dir`: file discovery, one-time conversion of legacy pickled `.dat`
  arrays to `.npy` with float64 downcast to float32, skipping non-array pickles, never
  overwriting existing output, and returning memory-mapped features with flat labels.

### `test_dataset_perf.py`

Two `pytest-benchmark` timings of the memory-mapped access path, marked `perf` and
therefore skipped in the default run. One measures per-row random access through
`IndexedArrayDataset`, including cross-array index resolution. The other measures the
bulk label gather in `get_split_labels` used at split time. Run them with
`python3.12 -m pytest -m perf` when you change the data layer and want before/after
numbers.

### `test_client_helpers.py`

Tests for two small helpers in `client.py`. `empty_evaluate_res` must return a Flower
evaluation result with zero loss, zero examples, and zeroed metrics, which the client
sends when it has nothing to evaluate. `loader_to_dmatrix` must concatenate the batches
yielded by a data loader into a single XGBoost `DMatrix` and flatten `(n, 1)` label
arrays to one dimension.

### `test_server_helpers.py`

Tests for the aggregation side in `server.py`:

- `weighted_average_metrics` weights each client's metrics by its example count, drops
  clients with zero examples or empty metric dicts, and treats a metric missing from
  one client as zero for that client.
- `get_torch_model_class` returns the DP-CNN class for `dpcnn`, and the plain CNN class
  for `cnn` or any unrecognized string.
- `update_global_history` appends per-round loss, accuracy, and MSE for every problem
  type, and appends precision and recall only for classification. A fixture snapshots
  and restores the module-level history lists so tests do not leak state.

### `test_model_metrics.py`

Tests for the metric code shared by every model in `model.py`. `BaseModel.task.metrics`
must produce accuracy, MSE, and MAE for regression, with accuracy defined by a tolerance
band, and accuracy plus macro precision and recall for classification.
`add_epoch_report_metrics` must include only the keys relevant to the problem type.
Two tests substitute a stub booster into `XGBoostModel` to confirm that its
`dataset_metrics` method routes through the same shared metric functions for both
problem types.

### `test_reports.py`

Tests for `report.Report`, which writes the JSON summary of a run. The constructor must
reject non-dictionary input. Plain dictionaries round-trip through `save_to_file`. NumPy
integers, floats, and arrays are converted to native JSON types. Two characterization
tests document current behavior rather than desired behavior: an unserializable object
is written as `null` instead of raising, and `np.bool_` also becomes `null` rather than a
JSON boolean. If either is fixed, update the test deliberately.

### `test_dp_privacy.py`

These tests guard the privacy accounting contract and should be read alongside
`docs/privacy_accounting.md`. The DP-CNN path in `model.py` deliberately does not
customize the Renyi order grid used by the Opacus RDP accountant. It lets Opacus
calibrate the noise multiplier and report epsilon over its built-in `DEFAULT_ALPHAS`.
The tests pin that contract from several directions:

- **No degenerate orders.** Every default alpha is greater than one, and a grid that
  starts at alpha equal to one raises a `ZeroDivisionError` inside the RDP-to-DP
  conversion. A previous no-op override in `model.py` built exactly such a grid.
- **Instance attributes are ignored.** Setting an `alphas` attribute on the accountant
  does not change the reported epsilon, which is why removing that override was
  behavior-preserving.
- **The production path is clean.** `attach_privacy_engine` does not inject an `alphas`
  attribute and uses the `rdp` mechanism.
- **Tightness is bounded.** At the shipped configuration the optimal order sits on the
  top edge of the default grid, and Opacus warns about it. The tests assert that warning
  on purpose, then widen the grid and check that the reported epsilon drops by less
  than a documented slack threshold and that the calibrated noise multiplier does not
  change at all. A moderate-privacy contrast case confirms the boundary effect is
  specific to very strong privacy settings.

These tests use a synthetic accountant history rather than real data, so they run in
seconds. If you change any DP parameter in `configs/`, rerun this file and read
`docs/privacy_accounting.md` before interpreting a failure.

## Conventions for adding tests

- Match the style of the file you are editing: a section banner comment per function
  under test, one behavior per test, and `pytest.approx` for floating point.
- Prefer the fixtures in `conftest.py` over new on-disk data. Never add real genotype or
  phenotype data to the repository.
- Mark long tests `slow`, cross-module tests `integration`, and benchmarks `perf`.
- When you find surprising behavior, add a characterization test that documents it and
  say so in a comment, rather than silently working around it.
- Changes to `model.py` are privacy-critical. Any test that touches DP behavior should
  state how the (epsilon, delta) guarantee is preserved, as `test_dp_privacy.py` does.

---

This document was written with the assistance of Claude Code (Anthropic, model Claude
Fable 5.1). The assistant read each test script and drafted these descriptions in
accordance with the author's instructions. All content has been reviewed and verified
by the authors to ensure accuracy and originality.
