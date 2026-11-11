from pathlib import Path

import numpy as np

from preprocess import CLASSES, IMG_SIZE
from train import DATA

PER_CLASS = 1000
SEED = 2026
OUT = DATA.parent / "quickdraw_vehicles_fresh_test.npz"
DATASET = Path(__file__).resolve().parents[1] / "Dataset"


def main():
    old = np.load(DATA)
    X_old = np.concatenate([old["X_train"], old["X_val"], old["X_test"]])
    y_old = np.concatenate([old["y_train"], old["y_val"], old["y_test"]])
    flat_old = X_old.reshape(len(X_old), -1)

    rng = np.random.default_rng(SEED)
    X_new, y_new = [], []
    for label, name in enumerate(CLASSES):
        used = {r.tobytes() for r in flat_old[y_old == label]}
        mm = np.load(DATASET / f"{name}.npy", mmap_mode="r")
        cand = np.sort(rng.permutation(len(mm))[:PER_CLASS * 2])
        rows = mm[cand]                                       # (2000, 784) uint8
        fresh = [j for j, r in enumerate(rows) if r.tobytes() not in used]
        keep = rng.permutation(fresh)[:PER_CLASS]             # shuffle, then take 1,000
        print(f"{name:>10s}: {len(rows) - len(fresh)} of {len(rows)} candidates already used")
        X_new.append(rows[keep].reshape(-1, IMG_SIZE, IMG_SIZE))
        y_new.append(np.full(len(keep), label, np.uint8))

    X_new, y_new = np.concatenate(X_new), np.concatenate(y_new)

    # final proof: not one fresh drawing may match anything we trained or validated on
    all_old = {r.tobytes() for r in flat_old}
    overlap = sum(r.tobytes() in all_old for r in X_new.reshape(len(X_new), -1))
    print(f"fresh test set {X_new.shape}, overlap with old splits: {overlap}")
    assert overlap == 0

    np.savez(OUT, X_test=X_new, y_test=y_new, classes=np.array(CLASSES))
    print("saved", OUT)


if __name__ == "__main__":
    main()