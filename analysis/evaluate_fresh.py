from pathlib import Path

import numpy as np

from losses import SoftmaxCrossEntropy
from model import build_baseline, build_cnn, load_weights
from train import DATA, WEIGHTS, evaluate

FRESH = DATA.parent / "quickdraw_vehicles_fresh_test.npz"

RUNS = [
    ("baseline",    build_baseline,           True,  WEIGHTS / "baseline_best.npz"),
    ("B",           lambda: build_cnn("B"),   False, WEIGHTS / "cnn_B_best.npz"),
    ("B + augment", lambda: build_cnn("B"),   False, WEIGHTS / "cnn_B_aug_best.npz"),
]


def main():
    d = np.load(FRESH)
    X, y = d["X_test"], d["y_test"].astype(np.int64)
    loss_fn = SoftmaxCrossEntropy()
    for name, build, flatten, path in RUNS:
        if not Path(path).exists():
            print(f"{name:12s} skipped (no {path})")
            continue
        model = load_weights(build(), path)
        loss, acc = evaluate(model, loss_fn, X, y, flatten)
        margin = 1.96 * np.sqrt(acc * (1 - acc) / len(X))
        print(f"{name:12s} | fresh-test loss {loss:.4f} acc {acc:.3f} +/- {margin:.3f}")


if __name__ == "__main__":
    main()