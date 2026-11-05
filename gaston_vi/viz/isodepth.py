"""Isodepth spatial plots.

Pure plotting: no RNG, no model state, no side effects beyond writing files.
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def plot_isodepth(S_xy, d_mu, save_path, title="Learned Isodepth",
                  edge_thresh_mult=3.0):
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(S_xy[:, 0], S_xy[:, 1],
                    c=d_mu, cmap="viridis",
                    s=0.6, linewidths=0, rasterized=True)

    triang = mtri.Triangulation(S_xy[:, 0], S_xy[:, 1])
    tri_pts = S_xy[triang.triangles]
    edges = np.stack([
        tri_pts[:, 1] - tri_pts[:, 0],
        tri_pts[:, 2] - tri_pts[:, 1],
        tri_pts[:, 0] - tri_pts[:, 2],
    ], axis=1)
    max_edge = np.linalg.norm(edges, axis=2).max(axis=1)
    triang.set_mask(max_edge > edge_thresh_mult * np.median(max_edge))

    ax.tricontour(triang,
                  np.asarray(d_mu).squeeze(),
                  levels=10, colors="black", linestyles="solid",
                  linewidths=0.8, zorder=10)   # solid: matplotlib dashes negative levels by default
    plt.colorbar(sc, ax=ax, label="isodepth")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_isodepth_per_batch(S_xy, d_mu, batch, save_dir, filename_prefix,
                             title_prefix="Isodepth"):
    """Plot isodepth separately for each batch.

    Subsets spots by batch label and calls plot_isodepth on each subset
    independently so contours are drawn within a single sample's tissue.
    """
    batch = np.asarray(batch).reshape(-1)
    for b in np.unique(batch):
        mask = (batch == b)
        if mask.sum() < 3:
            continue
        plot_isodepth(
            S_xy[mask], np.asarray(d_mu)[mask],
            save_path=f"{save_dir}/{filename_prefix}_batch{int(b)}.png",
            title=f"{title_prefix} — batch {int(b)}",
        )


def plot_all_isodepths(S_xy, d_mu_full, save_path, title_prefix="Isodepth"):
    M = d_mu_full.shape[1]
    ncols = min(4, M)
    nrows = math.ceil(M / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 3.5 * nrows),
                              squeeze=False)
    axes_flat = axes.flatten()
    for k in range(M):
        ax = axes_flat[k]
        sc = ax.scatter(S_xy[:, 0], S_xy[:, 1],
                        c=d_mu_full[:, k], cmap="viridis",
                        s=0.6, linewidths=0, rasterized=True)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"d_{k}", fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")
    for j in range(M, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(title_prefix, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
