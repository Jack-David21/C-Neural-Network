import numpy as np

CLASSES = ["train", "tractor", "speedboat", "car", "motorbike",
           "firetruck", "bulldozer", "ambulance", "airplane", "helicopter"]

for name in CLASSES:
    data = np.load(f"Dataset/{name}.npy", mmap_mode="r")
    print(f"{name:12s} shape={data.shape}  dtype={data.dtype}  min={data.min()}  max={data.max()}")