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

from pathlib import Path
import argparse
import pickle
import numpy as np

"""
This file converts .dat files into .npy files for more efficient loading into shared memory. This will not overwrite the .dat files.
Instead, this will create new .npy files with the same file prefixes as the .dat files. 
"""

def convert_dat_to_npy(data_dir: Path) -> None:
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {data_dir}")

    dat_files = sorted(data_dir.glob("*.dat"))
    npy_files = sorted(data_dir.glob("*.npy"))

    if not dat_files and not npy_files:
        print(f"No .dat files or .npy files found in {data_dir}. Please add in .dat data for conversions.")
        return

    for dat_path in dat_files:
        out_path = dat_path.with_suffix(".npy")

        if out_path.exists():
            print(f"Already exists: {out_path.name}")
            continue

        with dat_path.open("rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, np.ndarray):
            print(f"Skipping {dat_path.name}: not a NumPy array ({type(obj)})")
            continue

        arr = obj.astype(np.float32, copy=False) if obj.dtype == np.float64 else obj

        np.save(out_path, arr)
        print(f"Converted: {dat_path.name} -> {out_path.name} {arr.shape} {arr.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert .dat pickle files containing NumPy arrays to .npy files."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing .dat files",
    )

    args = parser.parse_args()
    convert_dat_to_npy(args.data_dir)


if __name__ == "__main__":
    main()
