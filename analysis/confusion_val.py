import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from model import build_cnn, build_cnn_stacked, load_weights
from preprocess import CLASSES
from train import DATA, WEIGHTS, prepare

FIGURES = Path(__file__).resolve().parents[1] / "figures"

MODELS = {
    "B": (lambda: build_cnn("B"), WEIGHTS / "cnn_B_best.npz"),
    "C": (build_cnn_stacked, WEIGHTS / "cnn_C_best.npz"),
    "Baug": (lambda: build_cnn("B"), WEIGHTS / "cnn_B_aug_best.npz"),
}
K = len(CLASSES)


def predict(model, X, batch_size=512):
    out = []
    for i in range(0, len(X), batch_size):
        logits = model.forward(prepare(X[i:i + batch_size], flatten=False))
        out.append(logits.argmax(axis=1))
    return np.concatenate(out)


def confusion(key, X, y):
    build, path = MODELS[key]
    pred = predict(load_weights(build(), path), X)
    # cm[i, j] = how many drawings of true class i were predicted as class j
    return np.bincount(y * K + pred, minlength=K * K).reshape(K, K)


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "C"
    d = np.load(DATA)
    X_val, y_val = d["X_val"], d["y_val"].astype(np.int64)

    cm = confusion(key, X_val, y_val)
    print(f"model {key}: overall val acc {cm.trace() / cm.sum():.3f}\n")

    print(" " * 12 + " ".join(f"{c[:4]:>5s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>11s} " + " ".join(f"{v:5d}" for v in cm[i]))

    acc = cm.diagonal() / cm.sum(axis=1)
    if key != "B":
        cm_b = confusion("B", X_val, y_val)
        acc_b = cm_b.diagonal() / cm_b.sum(axis=1)
        print(f"\nper-class accuracy: B -> {key}")
        for i in np.argsort(acc - acc_b):
            print(f"  {CLASSES[i]:>10s} {acc_b[i]:.3f} -> {acc[i]:.3f}"
                  f"  ({(acc[i] - acc_b[i]) * 100:+.1f} points)")
    else:
        print("\nper-class accuracy")
        for i in np.argsort(acc):
            print(f"  {CLASSES[i]:>10s} {acc[i]:.3f}")

    off = cm.copy()
    np.fill_diagonal(off, 0)
    print("\nmost common mistakes (true -> predicted, count)")
    for t, p in zip(*np.unravel_index(np.argsort(off, axis=None)[::-1][:8], off.shape)):
        print(f"  {CLASSES[t]:>10s} -> {CLASSES[p]:<10s} {off[t, p]:4d}")

    # heatmap: colour shows only the mistakes (diagonal zeroed), numbers are the real counts
    shade = cm.astype(float)
    np.fill_diagonal(shade, 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(shade, cmap="Blues")
    ax.set_xticks(range(K), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(K), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if shade[i, j] > shade.max() / 2 else "black")
    fig.colorbar(im, label="mistakes")
    fig.tight_layout()
    path = FIGURES / f"confusion_val_{key}.png"
    fig.savefig(path, dpi=150)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()