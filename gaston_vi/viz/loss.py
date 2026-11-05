"""Training-curve plots.

Pure plotting: no RNG, no model state, no side effects beyond writing files.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curve(train_history, val_history, save_path, title="Loss",
                    warmup_epochs=0):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_history, linewidth=2, color="steelblue", label="train recon loss")
    if val_history is not None:
        ax.plot(val_history, linewidth=2, color="tomato",
                linestyle="--", label="val recon loss")
    if warmup_epochs > 0:
        ax.axvline(x=warmup_epochs, color="gray", linestyle=":", linewidth=1.5,
                   label=f"phase 2 start (epoch {warmup_epochs})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("NB NLL (recon loss)")
    ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
