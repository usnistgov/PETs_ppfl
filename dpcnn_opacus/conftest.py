def pytest_addoption(parser):
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run t46 end-to-end smoke test. Skipped by default.",
    )