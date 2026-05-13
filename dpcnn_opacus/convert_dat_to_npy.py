from pathlib import Path
import pickle
import numpy as np

data_dir = Path("../data/Oil_binned5")  # change this

for dat_path in data_dir.glob("*.dat"):
    with dat_path.open("rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, np.ndarray):
        print(f"Skipping {dat_path.name}: not a NumPy array ({type(obj)})")
        continue

    arr = obj.astype(np.float32, copy=False) if obj.dtype == np.float64 else obj
    out_path = dat_path.with_suffix(".npy")

    np.save(out_path, arr)
    print(f"{dat_path.name} -> {out_path.name} {arr.shape} {arr.dtype}")