## Memory-mapped data loading

The Flower simulation path uses memory-mapped `.npy` files for the DPCNN data arrays. This was added to reduce Ray out-of-memory failures caused by each client process loading and copying large genomics arrays.

### Previous behavior

The original Flower/Ray path loaded pickled `.dat` arrays inside each client process. It then built additional NumPy arrays such as:

- `vcf = np.concatenate((tt_vcf, ho_vcf), axis=0)`
- `pheno = np.concatenate((tt_pheno, ho_pheno), axis=0)`
- `combined_dataset = np.concatenate((vcf, pheno), axis=1)`
- client train/test slices such as `combined_dataset[train_indices]`

Those operations create full in-memory copies. With multiple Ray clients, the same large dataset could be loaded and copied several times at once.

### Current behavior

The federated client/server path now uses `load_npy_feature_label_data`, which:

- checks for required `.npy` files matching `_tt_vcf`, `_tt_pheno`, `_ho_vcf`, and `_ho_pheno`
- automatically converts missing `.npy` files from the matching `.dat` files
- opens the `.npy` arrays with `np.load(..., mmap_mode="r")`
- keeps the arrays file-backed instead of eagerly loading each full array into every Ray process

Using `mmap_mode="r"` means NumPy creates an array-like view over the `.npy` file instead of immediately copying the whole file into process memory. The operating system loads pages from the file only as rows are accessed. Since Ray runs clients in separate worker processes, this is important: multiple workers can map the same read-only data files without each worker eagerly owning a separate full private copy of every array.

This does not make the dataset free. Rows that are actively read still occupy memory, and PyTorch/Opacus still allocate tensors, gradients, optimizer state, and batch data during training. The benefit is that baseline dataset storage is file-backed and shared more efficiently by the OS, so memory usage is driven more by active training work and less by repeated full dataset copies in each Ray client.

The large `tt` and `ho` feature/label arrays stay physically separate:

- `tt_vcf`
- `tt_pheno`
- `ho_vcf`
- `ho_pheno`

The `IndexedArrayDataset` class treats those separate arrays as one logical `tt + ho` dataset. Global row indices keep their original meaning: rows `0..len(tt)-1` refer to `tt`, and later rows refer to `ho` after subtracting `len(tt)`.

This preserves the original partition-file behavior without building large concatenated arrays.

### Automatic conversion

If one or more required `.npy` files are missing, `dataset.py` calls `convert_dat_to_npy.convert_dat_to_npy(data_dir)` automatically. Existing `.npy` files are left in place, so conversion should only happen when needed.

The converter only converts `.dat` files that contain NumPy arrays. Non-array pickle files are skipped.

### Expected memory behavior

This change reduces memory by avoiding repeated full dataset copies across Ray client workers. It does not eliminate all memory use. Training can still use several GB of RAM because PyTorch, Opacus, Ray actors, optimizer state, gradients, and active batches all allocate memory.

A moderate RAM peak during training is expected. The important improvement is that memory should no longer scale as badly with repeated dataset copies per client. If running into OOM issues, try increasing the number of cpus allocated for each Ray/Flwr client (increase the num_cpus parameter from the command line or config.json). This will reduce the number of clients running at any given time and therefore reduce the overall RAM usage.

### Legacy path

The legacy `load_pickle_data` function remains available for older scripts such as centralized training. The federated Flower client/server path should use the mmap-backed loader instead.