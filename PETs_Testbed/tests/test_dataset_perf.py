# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software
#
# This file was edited with the assistance of Claude Code (Anthropic, model
# Claude Opus 4.8). The assistant proposed and wrote performance benchmarks for
# the memory-mapped data loading path in accordance with the author's
# instructions. All content has been reviewed and verified by the authors.
#
# These benchmarks are deselected from the default test run (see pytest.ini's
# `-m "not perf"`). Run them explicitly with:
#     python3.12 -m pytest -m perf

import numpy as np
import pytest

from dataset import IndexedArrayDataset, get_split_labels

pytestmark = pytest.mark.perf


@pytest.fixture
def large_backed_arrays():
    """Two mmap-backed feature/label arrays large enough to time meaningfully."""
    rng = np.random.default_rng(0)
    features_a = rng.random((20_000, 200), dtype=np.float64).astype(np.float32)
    features_b = rng.random((20_000, 200), dtype=np.float64).astype(np.float32)
    labels_a = rng.integers(0, 2, size=20_000).astype(np.float32)
    labels_b = rng.integers(0, 2, size=20_000).astype(np.float32)
    return [features_a, features_b], [labels_a, labels_b]


def test_indexed_dataset_random_access_throughput(benchmark, large_backed_arrays):
    """Benchmark per-row random access, including cross-array index resolution."""
    features_list, labels_list = large_backed_arrays
    total = sum(len(f) for f in features_list)
    rng = np.random.default_rng(1)
    indices = rng.permutation(total)
    ds = IndexedArrayDataset(features_list, labels_list, indices=indices)

    sample_positions = rng.integers(0, len(ds), size=2_000)

    def access():
        for pos in sample_positions:
            ds[int(pos)]

    benchmark(access)


def test_get_split_labels_gather_throughput(benchmark, large_backed_arrays):
    """Benchmark bulk label gather across backing arrays (used at split time)."""
    _, labels_list = large_backed_arrays
    total = sum(len(a) for a in labels_list)
    rng = np.random.default_rng(2)
    indices = rng.permutation(total)

    benchmark(get_split_labels, labels_list, indices)
