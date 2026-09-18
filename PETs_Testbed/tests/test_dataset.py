# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software
#
# This file was edited with the assistance of Claude Code (Anthropic, model
# Claude Opus 4.8). The assistant proposed and wrote unit tests for the data
# handling logic in dataset.py in accordance with the author's instructions.
# All content, scientific claims, and conclusions have been reviewed and
# verified by the authors to ensure accuracy and originality.

import numpy as np
import pytest

from dataset import (
    IndexedArrayDataset,
    build_label_to_index,
    convert_dat_to_npy,
    find_single_npy,
    get_split_labels,
    load_npy_feature_label_data,
    normalize_label,
    resolve_data_dir,
    train_test_indices_split,
    train_test_split_backed_indices,
)


# --------------------------------------------------------------------------- #
# normalize_label / build_label_to_index                                      #
# --------------------------------------------------------------------------- #
def test_normalize_label_unwraps_numpy_scalars():
    assert normalize_label(np.int64(5)) == 5
    assert isinstance(normalize_label(np.int64(5)), int)
    # Plain Python values pass through unchanged.
    assert normalize_label(7) == 7
    assert normalize_label("a") == "a"


def test_build_label_to_index_none_returns_none():
    assert build_label_to_index(None) is None


def test_build_label_to_index_maps_labels_to_positions():
    mapping = build_label_to_index([10, 20, 30])
    assert mapping == {10: 0, 20: 1, 30: 2}


def test_build_label_to_index_normalizes_numpy_keys():
    mapping = build_label_to_index([np.int64(10), np.int64(20)])
    # Keys must be plain ints so runtime label lookups succeed.
    assert mapping == {10: 0, 20: 1}
    assert all(isinstance(k, int) for k in mapping)


# --------------------------------------------------------------------------- #
# IndexedArrayDataset                                                          #
# --------------------------------------------------------------------------- #
def test_indexed_dataset_len_matches_indices(synthetic_split):
    ds = IndexedArrayDataset(
        synthetic_split["features_list"],
        synthetic_split["labels_list"],
        indices=[0, 2, 4],
    )
    assert len(ds) == 3


def test_indexed_dataset_getitem_resolves_across_backing_arrays(synthetic_split):
    # Indices span both backing arrays, including the boundary rows 2 and 3.
    indices = [0, 1, 2, 3, 4]
    ds = IndexedArrayDataset(
        synthetic_split["features_list"],
        synthetic_split["labels_list"],
        indices=indices,
    )
    for position, global_row in enumerate(indices):
        features, label = ds[position]
        assert features[0] == synthetic_split["expected_feature0"][global_row]
        assert label == synthetic_split["expected_labels"][global_row]


def test_indexed_dataset_respects_index_order(synthetic_split):
    # A shuffled/subset index list must be honored positionally.
    ds = IndexedArrayDataset(
        synthetic_split["features_list"],
        synthetic_split["labels_list"],
        indices=[4, 0],
    )
    _, first_label = ds[0]
    _, second_label = ds[1]
    assert first_label == 14.0
    assert second_label == 10.0


def test_indexed_dataset_reshapes_2d_labels():
    features = [np.array([[1.0], [2.0]], dtype=np.float32)]
    labels = [np.array([[7.0], [8.0]], dtype=np.float32)]  # stored 2-D (n, 1)
    ds = IndexedArrayDataset(features, labels, indices=[0, 1])
    _, label = ds[1]
    assert np.ndim(label) == 0
    assert label == 8.0


def test_indexed_dataset_encodes_classification_labels():
    features = [np.array([[1.0], [2.0]], dtype=np.float32)]
    labels = [np.array([20.0, 10.0], dtype=np.float32)]
    label_to_index = build_label_to_index([10, 20])
    ds = IndexedArrayDataset(features, labels, indices=[0, 1], label_to_index=label_to_index)
    assert ds[0][1] == 1  # raw label 20 -> class index 1
    assert ds[1][1] == 0  # raw label 10 -> class index 0


def test_indexed_dataset_unknown_label_raises():
    features = [np.array([[1.0]], dtype=np.float32)]
    labels = [np.array([99.0], dtype=np.float32)]
    label_to_index = build_label_to_index([10, 20])
    ds = IndexedArrayDataset(features, labels, indices=[0], label_to_index=label_to_index)
    with pytest.raises(ValueError, match="not present in class_labels"):
        _ = ds[0]


def test_indexed_dataset_mismatched_list_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        IndexedArrayDataset(
            [np.zeros((2, 1), dtype=np.float32)],
            [np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)],
            indices=[0],
        )


def test_indexed_dataset_feature_label_length_mismatch_raises():
    with pytest.raises(ValueError, match="different lengths"):
        IndexedArrayDataset(
            [np.zeros((3, 1), dtype=np.float32)],
            [np.zeros(2, dtype=np.float32)],
            indices=[0],
        )


# --------------------------------------------------------------------------- #
# get_split_labels                                                            #
# --------------------------------------------------------------------------- #
def test_get_split_labels_gathers_across_arrays():
    labels = [np.array([10.0, 11.0, 12.0]), np.array([13.0, 14.0])]
    out = get_split_labels(labels, indices=[4, 0, 3])
    assert out.tolist() == [14.0, 10.0, 13.0]


@pytest.mark.parametrize("empty_indices", [[], np.array([], dtype=int), np.array([], dtype=float)])
def test_get_split_labels_empty_indices_returns_empty(empty_indices):
    # An empty index container of any dtype (plain list, int array, or the
    # float64 array NumPy infers from []) yields an empty result rather than
    # raising -- get_split_labels coerces indices to an integer dtype.
    labels = [np.array([1.0, 2.0])]
    out = get_split_labels(labels, indices=empty_indices)
    assert len(out) == 0


def test_get_split_labels_out_of_range_raises():
    labels = [np.array([1.0, 2.0]), np.array([3.0])]  # total logical size 3
    with pytest.raises(IndexError, match="outside logical dataset size"):
        get_split_labels(labels, indices=[3])


# --------------------------------------------------------------------------- #
# train_test_indices_split                                                    #
# --------------------------------------------------------------------------- #
def _classification_dataset(labels):
    """Build a 2-D dataset whose last column holds the given labels."""
    labels = np.asarray(labels, dtype=float)
    features = np.arange(len(labels), dtype=float).reshape(-1, 1)
    return np.hstack([features, labels.reshape(-1, 1)])


def test_train_test_split_is_disjoint_and_complete():
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    dataset = _classification_dataset(labels)
    indices = list(range(len(labels)))
    train, test = train_test_indices_split(dataset, indices, test_frac=0.2, seed=1,
                                            problem_type="classification")
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(indices)
    assert len(test) == 2  # 20% of 10 rows


def test_train_test_split_is_deterministic_under_seed():
    labels = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
    dataset = _classification_dataset(labels)
    indices = list(range(len(labels)))
    first = train_test_indices_split(dataset, indices, 0.3, seed=42,
                                     problem_type="classification")
    second = train_test_indices_split(dataset, indices, 0.3, seed=42,
                                      problem_type="classification")
    assert first == second


def test_train_test_split_puts_singleton_class_in_train():
    # Class label 2 occurs once: it cannot be stratified into the test split and
    # must be assigned to the training set.
    labels = [0, 0, 0, 1, 1, 1, 2]
    dataset = _classification_dataset(labels)
    indices = list(range(len(labels)))
    singleton_index = 6  # the lone label==2 row
    train, test = train_test_indices_split(dataset, indices, 0.2, seed=1,
                                           problem_type="classification")
    assert singleton_index in train
    assert singleton_index not in test


def test_train_test_split_regression_bins_and_splits():
    # Enough samples per quantile bin that stratified splitting is well-defined
    # (StratifiedShuffleSplit requires test_size >= number of populated bins).
    n = 100
    labels = np.linspace(0.0, 9.0, n)
    dataset = np.hstack([np.arange(n, dtype=float).reshape(-1, 1), labels.reshape(-1, 1)])
    indices = list(range(n))
    train, test = train_test_indices_split(dataset, indices, 0.25, seed=3,
                                           problem_type="regression")
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(indices)


def test_train_test_split_backed_indices_returns_global_indices():
    # Two backing label arrays; global indices offset into the second array.
    labels = [np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])]
    global_indices = [4, 5, 6, 7]  # all rows of the second backing array
    train, test = train_test_split_backed_indices(labels, global_indices, test_frac=0.5,
                                                  seed=1, problem_type="classification")
    combined = set(train) | set(test)
    assert combined.issubset(set(global_indices))
    assert set(train).isdisjoint(test)


# --------------------------------------------------------------------------- #
# find_single_npy / convert_dat_to_npy / load_npy_feature_label_data          #
# --------------------------------------------------------------------------- #
def test_find_single_npy_returns_match(npy_data_dir):
    path = find_single_npy(npy_data_dir["dir"], "tt_vcf")
    assert path is not None
    assert path.name.endswith("tt_vcf.npy")


def test_find_single_npy_returns_none_when_absent(tmp_path):
    assert find_single_npy(tmp_path, "tt_vcf") is None


def test_find_single_npy_multiple_matches_raise(tmp_path):
    np.save(tmp_path / "a_tt_vcf.npy", np.zeros((1, 1)))
    np.save(tmp_path / "b_tt_vcf.npy", np.zeros((1, 1)))
    with pytest.raises(ValueError, match="Multiple .npy files"):
        find_single_npy(tmp_path, "tt_vcf")


def test_convert_dat_to_npy_creates_files_and_downcasts(dat_data_dir):
    data_dir = dat_data_dir["dir"]
    convert_dat_to_npy(data_dir)

    vcf = np.load(data_dir / "S_tt_vcf.npy")
    np.testing.assert_array_equal(vcf, dat_data_dir["float64_arr"].astype(np.float32))
    # float64 inputs are stored as float32 to shrink the on-disk footprint.
    assert vcf.dtype == np.float32

    pheno = np.load(data_dir / "S_tt_pheno.npy")
    np.testing.assert_array_equal(pheno, dat_data_dir["int_arr"])


def test_convert_dat_to_npy_skips_non_array_pickles(dat_data_dir):
    data_dir = dat_data_dir["dir"]
    convert_dat_to_npy(data_dir)
    # The dict pickle must not produce a .npy file.
    assert not (data_dir / "S_notes.npy").exists()


def test_convert_dat_to_npy_does_not_overwrite_existing(dat_data_dir):
    data_dir = dat_data_dir["dir"]
    sentinel = np.array([[42.0]], dtype=np.float32)
    np.save(data_dir / "S_tt_vcf.npy", sentinel)

    convert_dat_to_npy(data_dir)

    # Existing .npy is preserved rather than regenerated from the .dat.
    np.testing.assert_array_equal(np.load(data_dir / "S_tt_vcf.npy"), sentinel)


def test_convert_dat_to_npy_bad_directory_raises(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(NotADirectoryError):
        convert_dat_to_npy(missing)


def test_load_npy_feature_label_data_returns_mmap_and_flat_labels(npy_data_dir):
    tt_vcf, tt_pheno, ho_vcf, ho_pheno, pub_vcf, pub_pheno = \
        load_npy_feature_label_data(npy_data_dir["dir"])

    np.testing.assert_array_equal(tt_vcf, npy_data_dir["features"]["tt_vcf"])
    # Phenotype labels are flattened to 1-D on load.
    assert tt_pheno.ndim == 1
    assert ho_pheno.ndim == 1
    assert pub_pheno.ndim == 1
    assert len(tt_pheno) == npy_data_dir["labels"]["tt_pheno"].shape[0]


def test_resolve_data_dir_keeps_absolute_paths(tmp_path):
    assert resolve_data_dir(tmp_path) == tmp_path.resolve()
