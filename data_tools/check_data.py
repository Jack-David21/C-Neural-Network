from pathlib import Path

import numpy as np

CLASSES = ["train", "tractor", "speedboat", "car", "motorbike",
           "firetruck", "bulldozer", "ambulance", "airplane", "helicopter"]

DATASET = Path(__file__).resolve().parents[1] / "Dataset"

for name in CLASSES:
    data = np.load(DATASET / f"{name}.npy", mmap_mode="r")
    print(f"{name:12s} shape={data.shape}  dtype={data.dtype}  min={data.min()}  max={data.max()}")