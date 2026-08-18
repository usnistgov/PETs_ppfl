## Regression and Unit Testing 

For regression testing, a Python library called `pytest` is used. This helps automate the testing process. There are various test cases described in the `tests/test_inputs.py` script. In this regression testing, it is only checking whether the parameter inputs are valid. To run the regression tests, ensure you are in the `PETs_Testbed` folder, and if using a virtual environment, make sure it is active. Also ensure you have `pytest` installed in your environment; it is now listed in `requirements.txt`. Then, run:
```bash
python3.10 -m pytest -q
``` 
This will loop through every single test case with a progress tracker at the bottom. Any failed tests will be printed at the end.

These regression tests do have an end-to-end run, but it is skipped by default due to the time it takes. If you want to include the end-to-end run in the regression testing, run:
```bash
python3.10 -m pytest -q --run-e2e
```

To manually go through each test case using `pytest`, first gather a list of all possible tests. I recommend recording it in a `.txt` file for easy lookup by running 
```bash
python3.10 -m pytest --collect-only -q > test_list.txt
```
Then, identify the test you want to run, for example `test_inputs.py::test_run_py_regressions[t12a]`. To run that individual test, use:

```bash
python3.10 -m pytest test_inputs.py::test_run_py_regressions[t12a]
```

### Adding Additional Regression Tests
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
