import sys
import time
from pathlib import Path

import numpy as np

from preprocess import to_model_input
from losses import SoftmaxCrossEntropy
from model import SGD, build_baseline, build_cnn, build_cnn_stacked

DATA = Path(__file__).parent / "Processed" / "quickdraw_vehicles_5000.npz"
WEIGHTS = Path(__file__).parent / "weights"


def prepare(xb, flatten=True):
    """(B, 28, 28) uint8 -> float32 in 0-1, converted per batch.
    flatten=True  -> (B, 784)       for the dense baseline
    flatten=False -> (B, 1, 28, 28) for the CNN
    """
    x = to_model_input(xb)  # (B, 1, 28, 28)
    return x.reshape(len(xb), -1) if flatten else x


def augment_batch(xb, rng, max_shift=2):
    """Random horizontal flip and random shift (up to max_shift px) per drawing.
    xb: (B, 28, 28) uint8. The background is 0, so zero padding is safe."""
    B, H, W = xb.shape
    out = xb.copy()
    flip = rng.random(B) < 0.5
    out[flip] = out[flip][:, :, ::-1]
    padded = np.pad(out, ((0, 0), (max_shift, max_shift), (max_shift, max_shift)))
    dy = rng.integers(0, 2 * max_shift + 1, B)
    dx = rng.integers(0, 2 * max_shift + 1, B)
    for i in range(B):
        out[i] = padded[i, dy[i]:dy[i] + H, dx[i]:dx[i] + W]
    return out


def evaluate(model, loss_fn, X, y, flatten=True, batch_size=512):
    total_loss, correct = 0.0, 0
    for i in range(0, len(X), batch_size):
        xb, yb = prepare(X[i:i + batch_size], flatten), y[i:i + batch_size]
        logits = model.forward(xb)
        total_loss += loss_fn.forward(logits, yb) * len(yb)
        correct += (logits.argmax(axis=1) == yb).sum()
    return total_loss / len(X), correct / len(X)


def train(build_fn=build_baseline, flatten=True, epochs=30, batch_size=64,
          lr=0.05, seed=0, data=DATA, save_path=None, augment=False):
    d = np.load(data)
    X_tr, y_tr = d["X_train"], d["y_train"].astype(np.int64)
    X_val, y_val = d["X_val"], d["y_val"].astype(np.int64)

    rng = np.random.default_rng(seed)
    model = build_fn(seed=seed)
    loss_fn = SoftmaxCrossEntropy()
    opt = SGD(model, lr)
    n = len(X_tr)

    best_loss, best_acc, best_epoch, best_weights = np.inf, 0.0, 0, None

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        run_loss, run_correct = 0.0, 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            if augment:
                xb = augment_batch(xb, rng)   # training only; validation stays untouched
            xb = prepare(xb, flatten)

            logits = model.forward(xb)
            loss = loss_fn.forward(logits, yb)
            model.backward(loss_fn.backward())
            opt.step()

            run_loss += loss * len(idx)
            run_correct += (logits.argmax(axis=1) == yb).sum()

        val_loss, val_acc = evaluate(model, loss_fn, X_val, y_val, flatten)

        improved = val_loss < best_loss
        if improved:
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            best_weights = [p.copy() for p, _ in model.params()]
            if save_path:
                np.savez(save_path, *best_weights)   # checkpoint: safe to stop at any time

        print(f"epoch {epoch:2d} | train loss {run_loss / n:.4f} acc {run_correct / n:.3f}"
              f" | val loss {val_loss:.4f} acc {val_acc:.3f}"
              f" | {time.perf_counter() - t0:.1f}s" + (" *" if improved else ""))

    for (p, _), w in zip(model.params(), best_weights):
        p[...] = w
    print(f"restored best epoch {best_epoch}: val loss {best_loss:.4f} acc {best_acc:.3f}")

    if save_path:
        np.savez(save_path, *best_weights)
    return model


if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    aug = len(sys.argv) > 4 and sys.argv[4] == "aug"
    tag = ("" if seed == 0 else f"_s{seed}") + ("_aug" if aug else "")
    if choice == "baseline":
        train(epochs=epochs, seed=seed, augment=aug,
              save_path=WEIGHTS / f"baseline{tag}_best.npz")
    elif choice in ("A", "B"):
        train(build_fn=lambda seed: build_cnn(choice, seed), flatten=False,
              epochs=epochs, seed=seed, augment=aug,
              save_path=WEIGHTS / f"cnn_{choice}{tag}_best.npz")
    elif choice == "C":
        train(build_fn=build_cnn_stacked, flatten=False,
              epochs=epochs, seed=seed, augment=aug,
              save_path=WEIGHTS / f"cnn_C{tag}_best.npz")
    else:
        raise SystemExit("usage: python train.py [baseline|A|B|C] [epochs] [seed] [aug]")