import argparse
from pathlib import Path

import numpy as np

CLASSES = ["train", "tractor", "speedboat", "car", "motorbike",
           "firetruck", "bulldozer", "ambulance", "airplane", "helicopter"]
IMG_SIZE = 28
SPLITS = ("train", "val", "test")


def split_sizes(per_class, train_frac, val_frac):
    
    n_train = round(per_class * train_frac)
    n_val = round(per_class * val_frac)
    n_test = per_class - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("per-class count / fractions leave an empty split")
    return {"train": n_train, "val": n_val, "test": n_test}


def sample_class(path, per_class, rng):

    mm = np.load(path, mmap_mode="r")                                
    if per_class > len(mm):
        raise ValueError(f"{path.name} has only {len(mm)} drawings; need {per_class}")
    idx = np.sort(rng.choice(len(mm), per_class, replace=False))     
    rows = mm[idx]                                                   
    rows = rows[rng.permutation(per_class)]                          
    return rows.reshape(-1, IMG_SIZE, IMG_SIZE)


def build(data_dir, per_class, train_frac, val_frac, seed):

    rng = np.random.default_rng(seed)
    sizes = split_sizes(per_class, train_frac, val_frac)
    k = len(CLASSES)

    X = {s: np.empty((k * n, IMG_SIZE, IMG_SIZE), np.uint8) for s, n in sizes.items()}
    y = {s: np.empty(k * n, np.uint8) for s, n in sizes.items()}

    train_slots = rng.permutation(k * sizes["train"])

    for label, name in enumerate(CLASSES):
        rows = sample_class(Path(data_dir) / f"{name}.npy", per_class, rng)
        start = 0
        for s in SPLITS:
            n = sizes[s]
            part = rows[start:start + n]
            start += n
            slots = train_slots[label * n:(label + 1) * n] if s == "train" \
                else slice(label * n, (label + 1) * n)
            X[s][slots] = part
            y[s][slots] = label
    return X, y


def save(out_path, X, y):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, classes=np.array(CLASSES),
             **{f"X_{s}": X[s] for s in SPLITS},
             **{f"y_{s}": y[s] for s in SPLITS})


def load_splits(path):
    """Load the saved arrays (uint8, small in RAM). Keys: X_train, y_train, X_val, ..., classes."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def to_model_input(batch):
    """(B, 28, 28) uint8 -> (B, 1, 28, 28) float32 in [0, 1]. Call once per mini-batch."""
    out = batch.astype(np.float32)
    out /= 255.0
    return out[:, None]                                           


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-class", type=int, default=3000)
    p.add_argument("--train-frac", type=float, default=0.75)
    p.add_argument("--val-frac", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="Dataset")
    p.add_argument("--out-dir", default="Processed")
    a = p.parse_args()

    X, y = build(a.data_dir, a.per_class, a.train_frac, a.val_frac, a.seed)
    out = Path(a.out_dir) / f"quickdraw_vehicles_{a.per_class}.npz"
    save(out, X, y)

    print(f"saved {out}")
    for s in SPLITS:
        counts = np.bincount(y[s], minlength=len(CLASSES))
        print(f"  {s:5s} X={X[s].shape}  per-class min/max = {counts.min()}/{counts.max()}")


if __name__ == "__main__":
    main()