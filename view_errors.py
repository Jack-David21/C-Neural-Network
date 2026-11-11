import numpy as np
import matplotlib.pyplot as plt

from confusion_val import predict
from evaluate_test import load_weights
from model import build_cnn
from preprocess import CLASSES
from train import DATA


def grid(X, idx, title, path, rows=3, cols=8):
    cmap = "gray" if np.median(X) > 127 else "gray_r"   # always draw dark strokes on light or the reverse consistently
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.3 + 0.6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx[:rows * cols]):
        ax.imshow(X[i], cmap=cmap, interpolation="nearest")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    d = np.load(DATA)
    X_val, y_val = d["X_val"], d["y_val"].astype(np.int64)
    model = load_weights(build_cnn("B"), "cnn_B_best.npz")
    pred = predict(model, X_val)
    cid = {c: i for i, c in enumerate(CLASSES)}

    for t, p in [("firetruck", "ambulance"), ("ambulance", "firetruck"), ("tractor", "bulldozer")]:
        idx = np.where((y_val == cid[t]) & (pred == cid[p]))[0]
        grid(X_val, idx, f"true {t}, predicted {p} ({len(idx)} in total)",
             f"errors_{t}_as_{p}.png")

    for t in ("firetruck", "ambulance"):
        idx = np.where((y_val == cid[t]) & (pred == cid[t]))[0]
        grid(X_val, idx, f"true {t}, predicted correctly", f"correct_{t}.png")
    print("saved 5 png files")


if __name__ == "__main__":
    main()