"""Decoder: rho = softmax( f([d, condition]) + W.[z, batch] ) for NB; for
Gaussian, rho = f([d, condition]) + W.[z, batch] directly, no softmax.

`linear_hz` is the only decoder retained (323/323 runs used it); the additive
and combined variants were deleted 2026-09-28. Keeping h LINEAR in z is what
makes f(d) a separable, interpretable zonation curve -- a nonlinear h can
absorb spatial signal, which is exactly what the `additive`/`combined`
variants did.

**apply_softmax (2026-10-06, ported from gaston_vi_codex_condf.py, the
original bespoke CODEX script):** CODEX protein intensities are NOT
compositional data the way counts are -- there is no real "fixed total
protein budget" one channel's rise has to be balanced by another's fall the
way sequencing depth forces a real constraint on count data. Forcing softmax
onto 49 independently-scaled [0,1] intensities confirmed to produce a real
artifact (2026-10 session): one dominant marker drags every other marker's
f(d) curve down with it via the sum-to-1 constraint, regardless of that
marker's own biology -- visible as several unrelated crypt/villus markers
collapsing onto nearly the same curve shape. gaston_vi_codex_condf.py's own
decoder docstring already said "CODEX: returns raw values -- no softmax" --
this was a known-correct design choice in the original script that the
modularized package's "minimal adjustments" port never carried over.
`apply_softmax` defaults to True (unchanged NB behaviour); GASTON_VI sets it
to `likelihood == "nb"`.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class LinearBiologicalDecoder(nn.Module):
    """ρ = softmax( g([d, onehot(condition)]) + W·[z, onehot(batch)] ).

    Batch is concatenated to the linear h branch's input.  When n_conditions>0
    the condition one-hot (derived from batch via batch_to_condition) is
    concatenated to g's input, so the spatial program differs per condition;
    when n_conditions==0, g(d) stays pure spatial.  linear_z_importance
    inspects W[:, :z_dim] (the z columns).
    """
    def __init__(self, d_dim, z_dim, g_dim, hidden_dim=128, n_batches=0,
                 n_conditions=0, batch_to_condition=None, apply_softmax=True):
        super().__init__()
        self.z_dim = z_dim
        self.n_batches = n_batches
        self.n_conditions = n_conditions
        self.apply_softmax = apply_softmax
        if batch_to_condition is not None:
            self.register_buffer(
                "batch_to_condition",
                torch.as_tensor(batch_to_condition, dtype=torch.long))
        else:
            self.batch_to_condition = None

        self.f_net = nn.Sequential(
            nn.Linear(d_dim + n_conditions, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.f_out = nn.Linear(hidden_dim, g_dim)
        self.h_linear = (nn.Linear(z_dim + n_batches, g_dim, bias=False)
                        if z_dim > 0 else None)

    def _z_input(self, z, batch):
        if self.n_batches == 0:
            return z
        if batch is None:
            b_oh = z.new_zeros(z.shape[0], self.n_batches)
        else:
            b_oh = F.one_hot(batch.view(-1).long(),
                             num_classes=self.n_batches).to(z.dtype)
        return torch.cat([z, b_oh], dim=-1)

    def _f_input(self, d, batch=None, condition=None):
        """Concatenate the condition one-hot to d for the f branch.

        condition (explicit) takes precedence over batch: pass a python int /
        0-d tensor to evaluate f for one condition over a whole d grid, or a
        per-row tensor. Otherwise the condition is looked up from batch via
        batch_to_condition (the path used during training).
        """
        if self.n_conditions == 0:
            return d
        if condition is not None:
            if torch.is_tensor(condition):
                cond = condition.view(-1).long().to(d.device)
            else:
                cond = torch.full((d.shape[0],), int(condition),
                                  dtype=torch.long, device=d.device)
        elif batch is not None and self.batch_to_condition is not None:
            cond = self.batch_to_condition[batch.view(-1).long()]
        else:
            # No batch and no explicit condition: fall back to a zero one-hot.
            # The model never sees this during training (a real one-hot is always
            # fed), so for analysis always pass condition= or a real batch.
            return torch.cat(
                [d, d.new_zeros(d.shape[0], self.n_conditions)], dim=-1)
        c_oh = F.one_hot(cond, num_classes=self.n_conditions).to(d.dtype)
        return torch.cat([d, c_oh], dim=-1)

    def forward(self, d, z, batch=None):
        logits = self.f_out(self.f_net(self._f_input(d, batch)))
        if self.h_linear is not None and z.shape[-1] > 0:
            z_in = self._z_input(z, batch)
            logits = logits + self.h_linear(z_in)
        return torch.softmax(logits, dim=-1) if self.apply_softmax else logits

    def f_logits(self, d, batch=None, condition=None):
        return self.f_out(self.f_net(self._f_input(d, batch, condition)))

    def h_logits(self, z, batch=None):
        z_in = self._z_input(z, batch)
        return self.h_linear(z_in)

    def f_proportions(self, d, batch=None, condition=None):
        """Despite the name (kept for call-site compatibility), this returns
        raw f(d) values, not proportions, when apply_softmax=False."""
        logits = self.f_logits(d, batch, condition)
        return torch.softmax(logits, dim=-1) if self.apply_softmax else logits


def linear_z_importance(decoder, gene_names=None, save_path=None):
    # h_linear has shape (G, z_dim + n_batches); slice off the batch columns
    W_full = decoder.h_linear.weight.detach().cpu().numpy()
    W = W_full[:, :decoder.z_dim]
    importance = np.linalg.norm(W, axis=0)
    ranked = np.argsort(importance)[::-1]
    print("\n=== Linear h(z) latent importance (||W[:,k]||_2) ===")
    for rank, k in enumerate(ranked):
        print(f"  rank {rank+1:2d}  z_{k}  importance={importance[k]:.4f}")
        if gene_names is not None:
            top5 = np.argsort(np.abs(W[:, k]))[::-1][:5]
            print(f"    top genes: {[gene_names[g] for g in top5]}")
    if save_path:
        fig, ax = plt.subplots(figsize=(max(6, len(importance) * 0.6 + 2), 4))
        ax.bar(range(len(importance)), importance, edgecolor="k", linewidth=0.5)
        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels([f"z_{k}" for k in range(len(importance))],
                           rotation=45, ha="right")
        ax.set_ylabel("||W[:,k]||_2"); ax.set_title("Linear h(z) latent importance")
        ax.grid(axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return importance
