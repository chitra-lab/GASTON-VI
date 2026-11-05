"""Regularisation terms added to the ELBO.

Two independent penalties, both optional and both off by default:

`compute_disentanglement_reg` (--di_method / --di_weight) discourages the
biological latent z from re-encoding the isodepth d, so f(d) keeps the spatial
signal instead of it leaking into h(z). d is DETACHED, so the gradient only
reshapes z. Variants: simvi_mi (Gaussian closed-form MI), simvi_mmd (kernel
MMD between the joint and the shuffled product), and the _kl suffixes which add
a KL(q(z)||N(0,I)) term.

`compute_batch_d_mmd` (--batch_align_weight) pulls the per-batch marginal
distributions of d onto a COMMON SCALE. Each batch has its own spatial encoder,
so nothing otherwise ties their d axes together and cross-batch isodepth would
be incomparable. With batch_to_condition set, only same-condition batch pairs
are penalised -- which is why per-condition f(d) curves are NOT calibrated
against each other (see the note in train.py's docstring).
The caller restricts the optimiser to `_spatial_encoder_params(model)` so this
term updates the spatial encoders only.

NOTE: `_closed_form_mi` and `_kernel_mmd` draw from the global torch RNG
(randn_like / randperm), so these functions sit inside the seeded trajectory --
reordering the calls or adding a draw changes every downstream result. Verify
with pe_heuristic/regression.py after touching them.

Nothing here assumes counts, apart from the log1p in
compute_disentanglement_reg feeding the bio encoder.
"""
import torch


def _closed_form_mi(z_mu, z_sigma, d_mu, d_sigma):
    z = z_mu + z_sigma * torch.randn_like(z_mu)
    d = d_mu + d_sigma * torch.randn_like(d_mu)
    zd = torch.cat([z, d], dim=-1)
    B  = zd.shape[0]
    zd_c   = zd - zd.mean(0, keepdim=True)
    cov_zd = (zd_c.T @ zd_c) / max(B - 1, 1) + 1e-6 * torch.eye(
        zd.shape[1], device=zd.device)
    dz = z.shape[1]
    cov_z = cov_zd[:dz, :dz]
    cov_d = cov_zd[dz:, dz:]
    logdet_z  = torch.linalg.slogdet(cov_z)[1]
    logdet_d  = torch.linalg.slogdet(cov_d)[1]
    logdet_zd = torch.linalg.slogdet(cov_zd)[1]
    mi = 0.5 * (logdet_z + logdet_d - logdet_zd)
    return torch.clamp(mi, min=0.0)


def _kernel_mmd(z, d, kernel_bandwidth=1.0):
    zd_joint = torch.cat([z, d], dim=-1)
    idx = torch.randperm(d.shape[0], device=d.device)
    zd_prod = torch.cat([z, d[idx]], dim=-1)

    def _gaussian_kernel(x, y, bw):
        dist_sq = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(-1)
        return torch.exp(-dist_sq / (2.0 * bw ** 2)).mean()

    bw = kernel_bandwidth
    mmd_sq = (
        _gaussian_kernel(zd_joint, zd_joint, bw)
        + _gaussian_kernel(zd_prod,  zd_prod,  bw)
        - 2.0 * _gaussian_kernel(zd_joint, zd_prod, bw)
    )
    return torch.clamp(mmd_sq, min=0.0)


def _get_d_sigma(model, spatial_batch, batch_idx=None):
    if model.n_isodepth == 1:
        _, d_sigma = model.spatial_encoder(spatial_batch, batch_idx)
    else:
        d_sigma = torch.cat(
            [enc(spatial_batch, batch_idx)[1] for enc in model.spatial_encoders],
            dim=-1)
    return d_sigma


def _kl_to_standard_normal(mu, sigma):
    return 0.5 * (mu.pow(2) + sigma.pow(2) - 2 * sigma.log() - 1).sum(dim=-1)


def compute_disentanglement_reg(method, model, gene_batch, spatial_batch,
                                batch_idx=None):
    if method == "none" or model.z_dim == 0:
        return torch.tensor(0.0, device=gene_batch.device)

    x_log1p = torch.log1p(gene_batch)
    z_mu, z_sigma = model.bio_encoder(x_log1p, batch_idx)
    # only regularize z
    d_mu          = model.encode_d_mu(spatial_batch, batch_idx).detach()
    d_sigma       = _get_d_sigma(model, spatial_batch, batch_idx).detach()

    if method == "simvi_mi":
        return _closed_form_mi(z_mu, z_sigma, d_mu, d_sigma)

    elif method == "simvi_mmd":
        z = z_mu + z_sigma * torch.randn_like(z_mu)
        d = d_mu + d_sigma * torch.randn_like(d_mu)
        return _kernel_mmd(z, d)

    elif method == "simvi_mi_kl":
        mi = _closed_form_mi(z_mu, z_sigma, d_mu, d_sigma)
        kl = _kl_to_standard_normal(z_mu, z_sigma).mean()
        return mi + kl

    elif method == "simvi_mmd_kl":
        z = z_mu + z_sigma * torch.randn_like(z_mu)
        d = d_mu + d_sigma * torch.randn_like(d_mu)
        mmd = _kernel_mmd(z, d)
        kl  = _kl_to_standard_normal(z_mu, z_sigma).mean()
        return mmd + kl

    raise ValueError(f"Unknown di_method: {method}")


# ------------------------------------------------------------
# Cross-batch isodepth alignment (MMD)
# ------------------------------------------------------------
def _two_sample_mmd(a, b, bandwidths=(0.5, 1.0, 2.0, 4.0)):
    """Multi-scale RBF MMD^2 between two sample sets a (n, D) and b (m, D).

    Uses a sum of Gaussian kernels at several fixed bandwidths (robust to the
    unknown scale of d). Returns a non-negative scalar; 0 when the two
    empirical distributions coincide.
    """
    def _rbf_mean(x, y):
        d2 = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(-1)
        k = 0.0
        for bw in bandwidths:
            k = k + torch.exp(-d2 / (2.0 * bw ** 2))
        return k.mean()

    mmd_sq = _rbf_mean(a, a) + _rbf_mean(b, b) - 2.0 * _rbf_mean(a, b)
    return torch.clamp(mmd_sq, min=0.0)


def _spatial_encoder_params(model):
    """All parameters of the (per-batch) spatial encoders only."""
    params = []
    if model.n_isodepth == 1:
        params += list(model.spatial_encoder.parameters())
    else:
        for enc in model.spatial_encoders:
            params += list(enc.parameters())
    return params


def compute_batch_d_mmd(model, spatial_batch, batch_idx, min_per_batch=2,
                        batch_to_condition=None):
    """Mean pairwise RBF-MMD between the isodepth distributions of the batches
    present in this minibatch.

    Each spot's d is produced by its own batch's spatial encoder
    (model.encode_d_mu routes per batch), so minimising this term pulls every
    batch's marginal distribution of d onto a common scale. Gradients flow into
    all spatial encoders; the caller restricts the optimiser to those params so
    nothing else is touched.

    When batch_to_condition is provided (an array/sequence indexable by batch
    index, mapping each batch to a condition label), only pairs of batches that
    share the same condition contribute to the penalty. This aligns d across
    replicate batches within a condition (e.g. healthy vs diseased) without
    ever pulling the two conditions onto a common scale. When it is None, all
    batch pairs are aligned (original behaviour).
    """
    if model.n_batches < 2 or batch_idx is None:
        return spatial_batch.new_tensor(0.0)

    b_flat = batch_idx.view(-1)
    d_mu   = model.encode_d_mu(spatial_batch, batch_idx)   # (B, n_isodepth)

    groups, group_batches = [], []
    for b in range(model.n_batches):
        mask = (b_flat == b)
        if mask.sum().item() >= min_per_batch:
            groups.append(d_mu[mask])
            group_batches.append(b)

    if len(groups) < 2:
        return spatial_batch.new_tensor(0.0)

    total, n_pairs = spatial_batch.new_tensor(0.0), 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if batch_to_condition is not None and (
                batch_to_condition[group_batches[i]]
                != batch_to_condition[group_batches[j]]):
                continue  # skip cross-condition pairs
            total = total + _two_sample_mmd(groups[i], groups[j])
            n_pairs += 1
    return total / max(1, n_pairs)


# ------------------------------------------------------------
# Per-epoch isodepth projection (--project_d)
# ------------------------------------------------------------
# Ported from gaston_vi_codex_condf.py (the original bespoke CODEX script,
# codex/condition_batch/) -- NOT present in this package before this port.
# An explicit per-epoch re-standardization layered on top of whatever
# compute_batch_d_mmd's gradient-based alignment already did that epoch;
# unlike compute_batch_d_mmd (a soft loss term, gradients blend with
# everything else on the spatial-encoder optimiser step), this directly
# rewrites the final linear layer's weight/bias so d_mu has mean=0, std=1 over
# each batch's full training set, exactly, every epoch -- no @torch.no_grad()
# step is partial.
@torch.no_grad()
def project_spatial_encoders_epoch(model, S_train_by_batch, eps=1e-6):
    """Rescale each batch's SpatialEncoder final linear layer so d_mu has
    mean=0, std=1 over the full training set for that batch:
        d = W*h + b  =>  d_new = (W/sigma)*h + (b - mu)/sigma
    Requires a BatchedSpatialEncoder (model.spatial_encoder.encoders); no-op
    (same as the original) for the single-section / n_isodepth>1 cases.
    """
    if not (hasattr(model, 'spatial_encoder') and
            hasattr(model.spatial_encoder, 'encoders')):
        return
    for b, s_b in enumerate(S_train_by_batch):
        if s_b.shape[0] < 2:
            continue
        enc = model.spatial_encoder.encoders[b]
        r   = enc(s_b)[0].squeeze(-1)
        mu  = r.mean()
        std = r.std(unbiased=False).clamp_min(eps)
        enc.mu.weight.div_(std)
        enc.mu.bias.sub_(mu).div_(std)
