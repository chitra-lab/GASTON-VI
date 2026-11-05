"""z-latent spatial plots.

Pure plotting: no RNG, no model state, no side effects beyond writing files.
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_z_latents(S_xy, z_mu, d_mu_col0, save_path, title_prefix="z latents"):
    z_dim = z_mu.shape[1]
    if z_dim == 0:
        return

    ncols    = min(4, z_dim + 1)
    nrows    = math.ceil((z_dim + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 3.5 * nrows),
                              squeeze=False)
    axes_flat = axes.flatten()

    sc = axes_flat[0].scatter(S_xy[:, 0], S_xy[:, 1],
                               c=d_mu_col0, cmap="viridis",
                               s=1.0, linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=axes_flat[0])
    axes_flat[0].set_title("isodepth d_0", fontsize=9)
    axes_flat[0].set_aspect("equal"); axes_flat[0].axis("off")

    for i in range(z_dim):
        ax = axes_flat[i + 1]
        sc = ax.scatter(S_xy[:, 0], S_xy[:, 1],
                        c=z_mu[:, i], cmap="viridis",
                        s=1.0, linewidths=0, rasterized=True)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"z_{i}", fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")

    for j in range(z_dim + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title_prefix, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_z_latents_per_batch(S_xy, z_mu, batch, save_dir, filename_prefix,
                              title_prefix="z latents"):
    """Plot all z dimensions in one multi-panel figure per batch."""
    z_dim = z_mu.shape[1]
    if z_dim == 0:
        return
    batch = np.asarray(batch).reshape(-1)
    for b in np.unique(batch):
        mask = (batch == b)
        if mask.sum() < 3:
            continue
        S_b = S_xy[mask]
        z_b = z_mu[mask]

        ncols = min(4, z_dim)
        nrows = math.ceil(z_dim / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(4 * ncols, 3.5 * nrows),
                                  squeeze=False)
        axes_flat = axes.flatten()
        for i in range(z_dim):
            ax = axes_flat[i]
            sc = ax.scatter(S_b[:, 0], S_b[:, 1],
                            c=z_b[:, i], cmap="viridis",
                            s=1.0, linewidths=0, rasterized=True)
            plt.colorbar(sc, ax=ax)
            ax.set_title(f"z_{i}", fontsize=9)
            ax.set_aspect("equal"); ax.axis("off")
        for j in range(z_dim, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle(f"{title_prefix} — batch {int(b)}", fontsize=12, y=1.01)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{filename_prefix}_batch{int(b)}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
