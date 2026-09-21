## Regression and Unit Testing 

The suite lives in `PETs_Testbed/tests/` and is split into three kinds of test:

| Kind | Files | What it checks |
| --- | --- | --- |
| Regression | `test_inputs.py` | Whole-program runs driven by config files and CLI overrides; validates that parameter inputs are accepted or rejected as expected. |
| Unit / characterization | `test_dataset.py`, `test_utils_config.py`, `test_server_helpers.py`, `test_client_helpers.py`, `test_reports.py`, `test_model_metrics.py`, `test_dp_privacy.py` | Individual functions in isolation, with no subprocess or full run. |
| Performance | `test_dataset_perf.py` | Benchmarks of the memory-mapped data access path. Deselected by default. |

The regression tests are documented immediately below; the unit and performance
tests are documented in [Unit, characterization, and performance tests](#unit-characterization-and-performance-tests).

### Regression Testing

For regression testing, a Python library called `pytest` is used. This helps automate the testing process. There are various test cases described in the `tests/test_inputs.py` script. In this regression testing, it is only checking whether the parameter inputs are valid. To run the regression tests, ensure you are in the `PETs_Testbed` folder, and if using a virtual environment, make sure it is active. Also ensure you have `pytest` installed in your environment; it is now listed in `requirements.txt`. Then, run:
```bash
python3.12 -m pytest -q
``` 
This will loop through every single test case with a progress tracker at the bottom. Any failed tests will be printed at the end.

These regression tests do have an end-to-end run, but it is skipped by default due to the time it takes. If you want to include the end-to-end run in the regression testing, run:
```bash
python3.12 -m pytest -q --run-e2e
```

To manually go through each test case using `pytest`, first gather a list of all possible tests. It can be recorded it in a `.txt` file for easy lookup by running 
```bash
python3.12 -m pytest --collect-only -q > test_list.txt
```
Then, identify the test you want to run, for example `test_inputs.py::test_run_py_regressions[t12a]`. To run that individual test, use:

```bash
python3.12 -m pytest test_inputs.py::test_run_py_regressions[t12a]
```

#### Adding Additional Regression Tests
Between the helper definitions near the top of `test_inputs.py` and the `CASES` list, the different regression test cases are defined. Each test is defined through a `Case()` instance and added to the list via `pytest.param()`. Helper functions at the top of the file make it easier to modify the parameters used in the regression tests and compare expected versus actual output. To modify the DPCNN parameters for a regression test, use the `_base_with()` helper function. To modify the CNN parameters for a regression test, use the `_cnn_base_with()` helper function. To modify the XGBoost parameters for a regression test, use the `_xgb_base_with()` helper function.

Here is the class definition of `Case()`:

```python
class Case:
    name: str
    config: Optional[Dict[str, Any]] = None
    raw_config_text: Optional[str] = None
    create_config_file: bool = True
    config_filename: str = "config.json"
    cli_args: Sequence[str] = field(default_factory=list)
    allowed_exit_codes: Set[int] = field(default_factory=lambda: {0})
    stdout_must_match: Sequence[str] = field(default_factory=list)
    stdout_must_not_match: Sequence[str] = field(default_factory=list)
```

The most important pieces of information when populating `Case()` are:

1. The name. This is a descriptor that allows someone looking through the regression test code, or the output of the regression tests, to know what the test is trying to accomplish.
2. The config. This is usually passed in using `_base_with()`, `_cnn_base_with()`, or `_xgb_base_with()`. `raw_config_text` should be used if you want to provide a full JSON file as a string to the test, for example if you have a very specific configuration file you want to test that is not easily expressible with the helper functions. These configuration files are typically written to a temporary pytest directory such as `/tmp/pytest-of-[username]/pytest-[num]/test_run_py_regressions_[test_id]`.
3. The command-line arguments. These can be helpful for testing overrides.
4. The acceptable exit codes. This tells the test whether you expect this case to fail or not. If you expect the regression test to fail and it does fail with one of the provided error codes, then the overall regression test passes.
5. Match or non-match output expectations. This is helpful when checking error handling. You can specify what you expect to see, or not see, in the output.

Please also note that `pytest.param()` expects a `Case()` instance and an `id` string to be provided.

Here are two examples of regression tests with the minimum information that should be provided. The first is an example of a regression test that is expected to fail, with an error message that the output must contain, and the second is an example of a regression test that is expected to pass:

```python
pytest.param(
    Case(
        "35. partition_id must be < num_partitions (if enforced)",
        _base_with(num_partitions=10, partition_id=11),
        allowed_exit_codes={1, 2},
        stdout_must_match=[r"<=|num_partitions"]
    ),
    id="t35"
),
pytest.param(
    Case(
        "36. client_id uniqueness in local sim (if enforced)",
        _base_with(client_id=1),
        cli_args=["--client_id", "1"],
        allowed_exit_codes={0},
    ),
    id="t36"
)
```

---

## Unit, characterization, and performance tests

The regression tests in `test_inputs.py` exercise the program end to end through
its CLI, which makes them good at catching broken configurations but slow and
coarse: a failure tells you a run exited non-zero, not which function is wrong.
The unit tests complement them by calling individual functions directly, with
small hand-built inputs and no subprocess.

Three of these files are **characterization tests** rather than specification
tests. A characterization test pins down what the code *currently* does, including
behavior that is arguably a bug, so that a future refactor cannot change it by
accident. Where a test documents a known gap rather than desired behavior, the
test body says so in a comment. Do not "fix" such a test by changing the
assertion — change the production code and the test together, deliberately.

### Test configuration (`pytest.ini`)

`PETs_Testbed/pytest.ini` holds the shared configuration:

- `testpaths = tests` — a bare `pytest` from `PETs_Testbed/` collects only the suite.
- Three markers are registered:
  - `perf` — performance benchmarks.
  - `slow` — long-running tests.
  - `integration` — tests spanning multiple modules or the filesystem.
- `addopts = -m "not perf"` — benchmarks are **deselected by default** so the
  everyday run stays fast. Passing `-m perf` on the command line overrides this.

### Running the tests

All commands assume you are in the `PETs_Testbed` folder with the virtual
environment active.

```bash
# Default run: unit + regression tests, benchmarks excluded.
python3.12 -m pytest -q

# One file.
python3.12 -m pytest tests/test_dataset.py

# One test.
python3.12 -m pytest tests/test_dataset.py::test_get_split_labels_out_of_range_raises

# Performance benchmarks only (overrides the default deselection).
python3.12 -m pytest -m perf

# Everything, including benchmarks.
python3.12 -m pytest -m ""

# Coverage for a module you are working on.
python3.12 -m pytest --cov=dataset --cov-report=term-missing
```

The expected default result on a clean checkout is `184 passed, 3 skipped,
2 deselected`. The 3 skips are the end-to-end regression cases (see
`--run-e2e` above); the 2 deselected are the performance benchmarks.

Coverage and benchmarking come from `pytest-cov` and `pytest-benchmark`, listed
in `requirements.txt`. Both are test-only dependencies and are not imported by
any production module.

### Shared fixtures (`tests/conftest.py`)

`conftest.py` provides the synthetic, file-backed data the unit tests use. None
of it touches real or sensitive data; every array is either hand-written or
generated from a seeded `np.random.default_rng`, so tests are deterministic.

| Fixture | Provides |
| --- | --- |
| `synthetic_split` | Two small feature arrays (3 and 2 rows) plus matching labels, along with the expected global row → label and row → first-feature values. The uneven lengths are deliberate: they force logical row indices to cross the boundary between backing arrays. |
| `npy_data_dir` | A `tmp_path` directory containing all six `.npy` files the loaders expect (`*_tt_vcf`, `*_ho_vcf`, `*_pub_vcf`, and the three matching `*_pheno` files). Features are 6×3 `float32`; labels are stored 2-D as `(n, 1)` to mirror the real on-disk phenotype layout. |
| `dat_data_dir` | A directory of pickled `.dat` arrays for conversion tests: one `float64` feature array, one `int64` label array, and one non-array pickle that the converter must skip. |

`conftest.py` also does two things at import time that are worth knowing about:

1. It registers the `--run-e2e` flag used by the regression tests.
2. It sets `OMP_NUM_THREADS=1`. On macOS, `torch` and `xgboost` each ship their
   own copy of `libomp`; importing `torch` before `xgboost` (as `client.py` does)
   can double-load OpenMP and segfault as soon as `xgboost` enters a parallel
   region, such as building a `DMatrix` in-process. Pinning the pool to one
   thread avoids that region. This affects only the test process — production
   code is untouched — but it does mean benchmark numbers are single-threaded.

### What each file covers

#### `test_dataset.py` — data handling (31 tests)

The largest unit file, covering `dataset.py`:

- **`normalize_label` / `build_label_to_index`** — NumPy scalar labels are
  unwrapped to plain Python `int`s, so that runtime dictionary lookups against
  the class-label map actually hit.
- **`IndexedArrayDataset`** — length matches the index list; `__getitem__`
  resolves a logical row index across multiple backing arrays (including the
  rows either side of the boundary); a shuffled or subset index list is honored
  positionally; 2-D `(n, 1)` labels are reshaped to scalars; classification
  labels are encoded to class indices; an unseen label raises `ValueError`; and
  mismatched feature/label list lengths are rejected at construction.
- **`get_split_labels`** — gathers labels across backing arrays in the requested
  order, returns empty for an empty index container of any dtype, and raises
  `IndexError` for an index past the logical dataset size.
- **`train_test_indices_split` / `train_test_split_backed_indices`** — the two
  splits are disjoint and jointly cover the input; the split is deterministic
  under a fixed seed; a singleton class that cannot be stratified lands in
  train; regression labels are quantile-binned before splitting; and the
  backed-array variant returns global rather than per-array indices.
- **`find_single_npy` / `convert_dat_to_npy` / `load_npy_feature_label_data` /
  `resolve_data_dir`** — discovery returns a single match or `None`, and raises
  on ambiguity; conversion downcasts `float64` to `float32`, skips non-array
  pickles, refuses to overwrite an existing `.npy`, and raises on a bad
  directory; loading flattens phenotype labels to 1-D.

The empty-index test exists because of a real fix: `get_split_labels` now
coerces indices to `np.intp`, so an empty list — which NumPy infers as `float64`
— no longer raises `IndexError`.

#### `test_utils_config.py` — the configuration engine (29 tests)

Direct tests of the schema-driven config machinery in `utils.py`, including its
private helpers:

- **`_coerce_cli_value`** — string CLI arguments are coerced per the schema type
  (boolean, integer, number, array, object, string), resolved through `oneOf`
  combinators; a value that cannot be coerced is passed through untouched so
  schema validation can report a precise field-level error instead of a generic
  parse failure.
- **`apply_defaults`** — fills scalar and nested defaults, never overrides a
  value the user supplied, and selects the correct `oneOf` branch using the
  `model_type` discriminator.
- **`_to_layered_config`** — flat keys are nested into their schema groups.
- **`_prune_inactive_model_params`** — parameters belonging to a different model
  type are dropped, shared and unknown parameters are preserved, and
  task-specific keys (for example `accuracy_tolerance`, which is
  regression-only) are removed for the wrong `problem_type`.
- **`_sync_enabled_flags`** — a group's `*_enabled` flag is inferred `True` when
  the group is populated and `False` when empty, but an explicitly supplied
  value always wins.
- **`_get_validation_errors` / `UnknownParameterError`** — missing required
  parameters, unknown properties, and type mismatches are normalized into
  `(path, value, message)` triples, and the "did you mean" suggestion appears
  only when there is a near match.
- **`validate_file_path` / `validate_dir_path` / `validate_data_size`** —
  missing paths raise `FileNotFoundError`; a batch size larger than the row
  count is rejected. The `.dat` fallback case covers a second real fix:
  `validate_data_size` now falls back to `.dat` feature files when no `.npy`
  exists, so batch-size validation is no longer silently skipped before
  conversion has run.

#### `test_server_helpers.py` — aggregation and history (7 tests)

- **`weighted_average_metrics`** — metrics are weighted by each client's example
  count; a zero total or an empty metric dict yields `{}` rather than a division
  error; a metric reported by only some clients is averaged as zero elsewhere.
- **`get_torch_model_class`** — `"dpcnn"` maps to `DPCNNModel`, and `"cnn"` and
  any unrecognized name map to `CNNModel`.
- **`update_global_history`** — regression rounds append to the loss, accuracy,
  and MSE histories and leave the classification histories untouched;
  classification rounds populate precision and recall. Because these histories
  are module-level lists, a `reset_history` fixture snapshots, clears, and
  restores them so the tests cannot leak state into each other.

#### `test_client_helpers.py` — client data conversion (3 tests)

- **`empty_evaluate_res`** — the zero-valued result returned when a client has
  nothing to evaluate.
- **`loader_to_dmatrix`** — batches of differing sizes concatenate into a single
  XGBoost `DMatrix` with the right row and column counts, and `(n, 1)` labels
  are flattened to a 1-D label vector.

#### `test_reports.py` — report serialization (5 tests, characterization)

- Non-dictionary input to `Report` raises `TypeError`.
- Plain dictionaries round-trip through JSON unchanged.
- NumPy scalars and arrays are serialized to JSON numbers and nested lists.
- **Known gap, pinned deliberately:** `NumpyEncoder.default` swallows the
  `TypeError` for an object it cannot encode and returns `None`, so the value is
  written as JSON `null` instead of the write failing loudly.
- **Known gap, pinned deliberately:** `np.bool_` is neither `np.integer` nor
  `np.floating`, so it falls through the same path and serializes as `null`
  rather than a JSON boolean.

The last two tests document current behavior, not desired behavior. If the
encoder is fixed, update these tests as part of that change.

#### `test_dp_privacy.py` — privacy mechanics (8 tests)

Per the testing philosophy in `CLAUDE.md`, these assert on privacy mechanics
rather than on model outputs. They guard the Opacus RDP accountant contract that
the DP-CNN training path in `model.py` depends on. `model.py` deliberately does
**not** customize the RDP order grid; it lets the accountant calibrate noise and
report (ε, δ) over its built-in `DEFAULT_ALPHAS`.

- No order in `RDPAccountant.DEFAULT_ALPHAS` is degenerate: the RDP-to-DP
  conversion divides by `(alpha - 1)`, so `alpha == 1` is a `ZeroDivisionError`
  and `alpha < 1` is not a valid Rényi order.
- A grid starting at `alpha == 1.0` — exactly the grid a previously removed
  override in `model.py` built — raises rather than reporting a budget, which is
  why it must never be wired into `get_privacy_spent`.
- Reported epsilon depends only on the default or explicitly passed grid, never
  on an instance `alphas` attribute. This is the property that made the removed
  override a no-op, and it must stay true for the removal to be
  behavior-preserving.
- The production path, `attach_privacy_engine`, does not inject a custom alpha
  grid onto the accountant, and the accountant's mechanism remains `"rdp"`.

If a future change reintroduces a custom alpha grid, these tests fail. That is
their purpose — treat a failure here as a privacy review item, not a test to
adjust.

A second group bounds how much tightness the default grid gives up. At the
shipped configuration Opacus warns that the optimal Rényi order is the largest
alpha, meaning the optimum sits on the top edge of `DEFAULT_ALPHAS` and a better
order may lie beyond it. That is a tightness question, not a soundness one — the
RDP-to-DP conversion minimizes over the grid, so a narrow grid can only overstate
epsilon. These tests pin the size of that effect:

- At the shipped regime the optimal order is `max(DEFAULT_ALPHAS) == 63` and the
  `UserWarning` fires. The warning is asserted with `pytest.warns` rather than
  suppressed, so the day it stops happening is a visible event.
- Widening the grid to `alpha = 128` must never report a *looser* bound, must
  place the optimum strictly inside the widened grid, and must not reduce epsilon
  by more than 5%. Measured slack today is 0.83% at the shipped configuration and
  1.08% at a lower sample rate. **This is the alarm:** if a future change to the
  DP parameters makes the default grid genuinely inadequate, it fails.
- The calibrated noise multiplier is identical with and without the widened grid
  (σ = 130.0), because the slack is far inside `get_noise_multiplier`'s default
  `epsilon_tolerance` of 0.01. This is why the looseness costs no model accuracy.
- At a moderate budget the optimum is interior (α = 10.7) and no warning fires,
  showing the boundary condition is specific to very strong privacy settings.

Background and full measurements are in
[Privacy accounting and the RDP order grid](privacy_accounting.md) and
`reports/2026-09-21-rdp-alpha-grid-tightness.md`.

#### `test_model_metrics.py` — metric computation (5 tests)

Pre-existing coverage of `BaseModel.task.metrics`, `add_epoch_report_metrics`,
and `XGBoostModel.dataset_metrics`, confirming that regression and
classification paths produce the metrics appropriate to each problem type and
that XGBoost reuses the shared metric implementations.

#### `test_dataset_perf.py` — benchmarks (2 tests, deselected by default)

Benchmarks of the memory-mapped data access path, marked with a module-level
`pytestmark = pytest.mark.perf`:

- **Random-access throughput** — per-row `__getitem__` on an
  `IndexedArrayDataset` over two 20,000 × 200 backing arrays with a shuffled
  index permutation, so every access also exercises cross-array index
  resolution.
- **Bulk gather throughput** — `get_split_labels` over a full permutation of
  40,000 logical rows, the operation performed at split time.

Both use the `benchmark` fixture from `pytest-benchmark`, which reports min,
max, mean, and standard deviation per run. Absolute numbers are
machine-dependent and single-threaded (see the `OMP_NUM_THREADS` note above), so
use them for before/after comparison on one machine rather than as a portable
target. To compare two runs:

```bash
python3.12 -m pytest -m perf --benchmark-autosave       # baseline
# ...make your change...
python3.12 -m pytest -m perf --benchmark-compare
```

### Adding unit tests

1. **Put the test next to its peers.** One file per production module,
   named `test_<module>.py`. Group related tests under a banner comment, as the
   existing files do.
2. **Prefer an existing fixture.** If you need file-backed data, use
   `npy_data_dir` or `dat_data_dir` rather than writing files inline. If a new
   shape of data is needed by more than one file, add a fixture to
   `conftest.py`; otherwise keep it local to the test module.
3. **Import the module under test directly** (`from dataset import ...`).
   Fixtures deliberately do not import production modules, so a broken import in
   the module under test surfaces as a failure in its own file rather than as a
   collection error across the suite.
4. **Assert on one behavior per test**, and name the test after that behavior —
   `test_get_split_labels_out_of_range_raises`, not `test_get_split_labels_2`.
5. **Mark appropriately.** Add `@pytest.mark.perf` to benchmarks (or a
   module-level `pytestmark`), `slow` to anything that takes more than a few
   seconds, and `integration` to tests that span modules or hit the filesystem.
   Unmarked tests run in the default suite, so anything unmarked should be fast.
6. **If you are pinning current behavior rather than specifying correct
   behavior, say so in a comment** with the word "characterization", so the next
   person knows the assertion is a description and not a requirement.
7. **Restore any global state you touch.** `test_server_helpers.py`'s
   `reset_history` fixture is the pattern to copy for module-level mutable state.
8. **Changes under `model.py` or anything affecting the DP guarantee need human
   review before implementation.** See the privacy-critical invariants section
   of `CLAUDE.md`.

---

*This documentation was written with the assistance of Claude Code (Anthropic,
model Claude Opus 5), which read the test suite and drafted these descriptions,
later updated the example commands from `python3.10` to `python3.12` as part of
the Python 3.12 migration, and later still documented the RDP order-grid
tightness tests, in accordance with the author's instructions. All content has
been reviewed and verified by the authors to ensure accuracy and originality.*
