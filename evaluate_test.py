import numpy as np

from losses import SoftmaxCrossEntropy
from model import build_baseline, build_cnn
from train import DATA, evaluate


def load_weights(model, path):
    """Copy the saved arrays (arr_0, arr_1, ...) into the model, in params() order."""
    with np.load(path) as f:
        saved = [f[f"arr_{i}"] for i in range(len(f.files))]
    params = model.params()
    assert len(saved) == len(params), "weight file does not match this model"
    for (p, _), w in zip(params, saved):
        assert p.shape == w.shape, "weight shape mismatch"
        p[...] = w
    return model


def main():
    d = np.load(DATA)
    X_val, y_val = d["X_val"], d["y_val"].astype(np.int64)
    X_te, y_te = d["X_test"], d["y_test"].astype(np.int64)
    loss_fn = SoftmaxCrossEntropy()

    runs = [
        ("baseline  ", build_baseline(), True, "baseline_best.npz"),
        ("CNN head B", build_cnn("B"), False, "cnn_B_best.npz"),
    ]

    for name, model, flatten, path in runs:
        load_weights(model, path)

        # check: must reproduce the validation numbers from the training log
        v_loss, v_acc = evaluate(model, loss_fn, X_val, y_val, flatten)

        # the one and only test evaluation for this model
        t_loss, t_acc = evaluate(model, loss_fn, X_te, y_te, flatten)
        margin = 1.96 * np.sqrt(t_acc * (1 - t_acc) / len(X_te))

        print(f"{name} | val loss {v_loss:.4f} acc {v_acc:.3f} (check)"
              f" | TEST loss {t_loss:.4f} acc {t_acc:.3f} +/- {margin:.3f}")


if __name__ == "__main__":
    main()