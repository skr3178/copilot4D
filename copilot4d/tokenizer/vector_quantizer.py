"""Vector Quantizer with EMA updates, straight-through gradients, K-Means init, and FP32 forcing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with EMA codebook updates.

    Encoder output -> LN -> GELU -> Linear(dim, codebook_dim) -> quantize -> Linear(codebook_dim, dim)

    Features:
    - Straight-through gradient estimator
    - EMA codebook updates
    - K-Means initialization
    - Dead code re-initialization
    - FP32 forced for numerical stability
    """

    def __init__(
        self,
        dim: int = 256,
        codebook_size: int = 1024,
        codebook_dim: int = 1024,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # Pre-VQ projection
        self.pre_norm = nn.LayerNorm(dim)
        self.pre_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, codebook_dim),
        )

        # Post-VQ projection
        self.post_proj = nn.Linear(codebook_dim, dim)

        # Codebook
        if kmeans_init:
            embed = torch.zeros(codebook_size, codebook_dim)
        else:
            embed = torch.randn(codebook_size, codebook_dim)

        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.ones(codebook_size))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("initted", torch.tensor(not kmeans_init))

    @torch.no_grad()
    def _init_embed(self, data: torch.Tensor):
        """Initialize codebook with K-Means."""
        if self.initted:
            return

        # data: (N, codebook_dim)
        embed, _ = _kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed)
        self.cluster_size.data.fill_(1.0)
        self.initted.fill_(True)

    @torch.no_grad()
    def _expire_codes(self, batch_samples: torch.Tensor):
        """Re-initialize dead codes from batch samples."""
        if self.threshold_ema_dead_code == 0:
            return

        expired = self.cluster_size < self.threshold_ema_dead_code
        if not expired.any():
            return

        n_expired = expired.sum().item()
        if batch_samples.shape[0] >= n_expired:
            indices = torch.randperm(batch_samples.shape[0], device=batch_samples.device)[:n_expired]
        else:
            indices = torch.randint(0, batch_samples.shape[0], (n_expired,), device=batch_samples.device)

        self.embed.data[expired] = batch_samples[indices]
        self.embed_avg.data[expired] = batch_samples[indices]
        self.cluster_size.data[expired] = self.threshold_ema_dead_code

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, dim) encoder output

        Returns:
            quantized: (B, N, dim)
            indices: (B, N)
            vq_loss: scalar
        """
        B, N, D = x.shape

        # Pre-projection (all in FP32)
        x_normed = self.pre_norm(x.float())
        z_e = self.pre_proj(x_normed)  # (B, N, codebook_dim)

        flat = z_e.reshape(-1, self.codebook_dim)  # (B*N, codebook_dim)

        # K-Means init on first batch
        if self.training:
            self._init_embed(flat)

        # Compute distances: ||z_e - e||^2 = ||z_e||^2 - 2*z_e*e^T + ||e||^2
        z_e_sq = (flat ** 2).sum(dim=1, keepdim=True)     # (B*N, 1)
        e_sq = (self.embed ** 2).sum(dim=1, keepdim=True)  # (K, 1)
        dist = z_e_sq - 2.0 * flat @ self.embed.t() + e_sq.t()  # (B*N, K)

        # Find nearest code
        indices = dist.argmin(dim=1)  # (B*N,)
        z_q = self.embed[indices]     # (B*N, codebook_dim)

        # EMA update
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices, self.codebook_size).float()  # (B*N, K)
                cluster_size = onehot.sum(0)                             # (K,)
                embed_sum = onehot.t() @ flat                            # (K, codebook_dim)

                self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                # Laplace smoothing
                n = self.cluster_size.sum()
                cluster_size_smoothed = (
                    (self.cluster_size + self.eps)
                    / (n + self.codebook_size * self.eps)
                    * n
                )
                self.embed.data.copy_(self.embed_avg / cluster_size_smoothed.unsqueeze(1))

                # Expire dead codes
                self._expire_codes(flat)

        # Commitment loss: ||z_e - sg(z_q)||^2
        commitment_loss = self.commitment_cost * F.mse_loss(flat, z_q.detach())

        # Straight-through estimator
        z_q_st = flat + (z_q - flat).detach()  # (B*N, codebook_dim)
        z_q_st = z_q_st.view(B, N, self.codebook_dim)

        # Post-projection back to dim
        quantized = self.post_proj(z_q_st)  # (B, N, dim)

        indices = indices.view(B, N)

        return quantized, indices, commitment_loss

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook entries and project back to model dim.

        Args:
            indices: (B, N) or (B, H, W) code indices

        Returns:
            (B, N, dim) or (B, H, W, dim)
        """
        shape = indices.shape
        flat_indices = indices.reshape(-1)
        z_q = self.embed[flat_indices]
        z_q = z_q.view(*shape, self.codebook_dim)
        return self.post_proj(z_q)


@torch.no_grad()
def _kmeans(
    data: torch.Tensor,
    num_clusters: int,
    num_iters: int = 10,
) -> tuple:
    """Simple K-Means clustering.

    Args:
        data: (N, D) samples
        num_clusters: K
        num_iters: number of iterations

    Returns:
        centers: (K, D)
        assignments: (N,)
    """
    N, D = data.shape
    # Random init
    if N >= num_clusters:
        indices = torch.randperm(N, device=data.device)[:num_clusters]
    else:
        indices = torch.randint(0, N, (num_clusters,), device=data.device)
    centers = data[indices].clone()

    for _ in range(num_iters):
        # Assign
        dists = torch.cdist(data, centers)  # (N, K)
        assignments = dists.argmin(dim=1)   # (N,)

        # Update
        for k in range(num_clusters):
            mask = assignments == k
            if mask.any():
                centers[k] = data[mask].mean(dim=0)

    return centers, assignments
