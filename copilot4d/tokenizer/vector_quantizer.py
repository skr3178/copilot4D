"""Vector Quantizer with K-Means codebook re-initialization from memory bank.

Following the CoPilot4D paper:
- Memory bank stores recent encoder outputs (size = 10 * codebook_size)
- Dead code: not used for 256 iterations
- Re-initialization: K-Means on memory bank if > 3% of codebook is dead
- Minimum 200 iterations before re-initialization
- Straight-through gradient estimator
- Loss: L_vq = 0.25 * ||sg[E(o)] - z_hat||^2 + 1.0 * ||sg[z_hat] - E(o)||^2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class VectorQuantizer(nn.Module):
    """Vector Quantization with memory bank and K-Means re-initialization.
    
    Paper implementation details:
    - Memory bank size = 10 * codebook_size
    - Dead code threshold = 256 iterations
    - Re-init if > 3% dead codes
    - Min 200 iterations before first re-init
    """

    def __init__(
        self,
        dim: int = 256,
        codebook_size: int = 1024,
        codebook_dim: int = 1024,
        commitment_cost: float = 0.25,  # lambda_1 in paper
        codebook_cost: float = 1.0,      # lambda_2 in paper
        kmeans_iters: int = 10,
        dead_threshold: int = 256,       # iterations before code is "dead"
        dead_percentage: float = 0.03,   # 3% threshold for re-init
        min_iterations: int = 200,       # min iterations before re-init
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost  # lambda_1 = 0.25
        self.codebook_cost = codebook_cost      # lambda_2 = 1.0
        self.kmeans_iters = kmeans_iters
        self.dead_threshold = dead_threshold    # 256 iterations
        self.dead_percentage = dead_percentage  # 0.03 = 3%
        self.min_iterations = min_iterations    # 200 iterations

        # Pre-VQ projection: encoder output -> codebook space
        self.pre_norm = nn.LayerNorm(dim)
        self.pre_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, codebook_dim),
        )

        # Post-VQ projection: codebook -> encoder output space
        self.post_proj = nn.Linear(codebook_dim, dim)

        # Codebook (initialized with K-Means later)
        self.register_buffer("embed", torch.randn(codebook_size, codebook_dim))
        
        # Usage counters for dead code detection
        self.register_buffer("usage_count", torch.zeros(codebook_size, dtype=torch.long))
        self.register_buffer("iteration_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("initted", torch.tensor(False))

        # Memory bank: stores recent encoder outputs
        memory_bank_size = 10 * codebook_size
        self.register_buffer("memory_bank", torch.zeros(memory_bank_size, codebook_dim))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _update_memory_bank(self, data: torch.Tensor):
        """Store encoder outputs in memory bank (circular buffer).
        
        Args:
            data: (N, codebook_dim) encoder outputs before quantization
        """
        batch_size = data.shape[0]
        ptr = int(self.bank_ptr)
        bank_size = self.memory_bank.shape[0]
        
        # Circular buffer: wrap around if needed
        if ptr + batch_size > bank_size:
            # Split into two parts
            first_part = bank_size - ptr
            self.memory_bank[ptr:] = data[:first_part]
            self.memory_bank[:batch_size - first_part] = data[first_part:]
            ptr = batch_size - first_part
        else:
            self.memory_bank[ptr:ptr + batch_size] = data
            ptr = (ptr + batch_size) % bank_size
            
        self.bank_ptr[0] = ptr

    @torch.no_grad()
    def _init_codebook_kmeans(self, data: torch.Tensor):
        """Initialize codebook with K-Means on first batch."""
        if self.initted:
            return
        
        embed, _ = _kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.initted.fill_(True)

    @torch.no_grad()
    def _check_and_reinit_dead_codes(self):
        """Check for dead codes and re-initialize with K-Means on memory bank.
        
        Re-initialization happens if:
        1. At least min_iterations (200) have passed
        2. More than dead_percentage (3%) of codes are dead
        """
        self.iteration_count += 1
        
        # Check minimum iterations
        if self.iteration_count < self.min_iterations:
            return
        
        # Find dead codes (not used for dead_threshold iterations)
        dead_codes = self.usage_count >= self.dead_threshold
        num_dead = dead_codes.sum().item()
        dead_ratio = num_dead / self.codebook_size
        
        # Re-initialize if > 3% dead
        if dead_ratio > self.dead_percentage:
            # Run K-Means on memory bank to re-initialize entire codebook
            bank_data = self.memory_bank[self.memory_bank.abs().sum(dim=1) > 0]  # Non-empty entries
            if bank_data.shape[0] >= self.codebook_size:
                new_embed, _ = _kmeans(bank_data, self.codebook_size, self.kmeans_iters)
                self.embed.data.copy_(new_embed)
                self.usage_count.zero_()  # Reset usage counters

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, dim) encoder output

        Returns:
            quantized: (B, N, dim)
            indices: (B, N)
            vq_loss: scalar (commitment + codebook loss)
        """
        B, N, D = x.shape

        # Pre-projection (all in FP32)
        x_normed = self.pre_norm(x.float())
        z_e = self.pre_proj(x_normed)  # (B, N, codebook_dim)

        flat = z_e.reshape(-1, self.codebook_dim)  # (B*N, codebook_dim)

        # K-Means init on first batch
        if self.training and not self.initted:
            self._init_codebook_kmeans(flat)

        # Update memory bank with encoder outputs
        if self.training:
            self._update_memory_bank(flat.detach())

        # Compute distances: ||z_e - e||^2
        z_e_sq = (flat ** 2).sum(dim=1, keepdim=True)
        e_sq = (self.embed ** 2).sum(dim=1, keepdim=True)
        dist = z_e_sq - 2.0 * flat @ self.embed.t() + e_sq.t()

        # Find nearest code
        indices = dist.argmin(dim=1)
        z_q = self.embed[indices]

        # Update usage counters
        if self.training:
            self.usage_count += 1  # Increment all
            used_codes = torch.unique(indices)
            self.usage_count[used_codes] = 0  # Reset used codes
            
            # Check for dead codes and re-initialize if needed
            self._check_and_reinit_dead_codes()

        # VQ Loss (paper: L_vq = 0.25 * ||sg[E(o)] - z_hat||^2 + 1.0 * ||sg[z_hat] - E(o)||^2)
        # Commitment loss: ||z_e - sg(z_q)||^2 (encoder should commit to codebook)
        commitment_loss = self.commitment_cost * F.mse_loss(flat, z_q.detach())
        
        # Codebook loss: ||sg(z_e) - z_q||^2 (codebook should move to encoder outputs)
        codebook_loss = self.codebook_cost * F.mse_loss(flat.detach(), z_q)
        
        vq_loss = commitment_loss + codebook_loss

        # Straight-through estimator
        z_q_st = flat + (z_q - flat).detach()
        z_q_st = z_q_st.view(B, N, self.codebook_dim)

        # Post-projection back to dim
        quantized = self.post_proj(z_q_st)
        indices = indices.view(B, N)

        return quantized, indices, vq_loss

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook entries and project back to model dim."""
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
    """K-Means clustering for codebook initialization.
    
    Args:
        data: (N, D) samples
        num_clusters: K
        num_iters: number of iterations

    Returns:
        centers: (K, D)
        assignments: (N,)
    """
    N, D = data.shape
    # Random init from data
    if N >= num_clusters:
        indices = torch.randperm(N, device=data.device)[:num_clusters]
    else:
        indices = torch.randint(0, N, (num_clusters,), device=data.device)
    centers = data[indices].clone()

    for _ in range(num_iters):
        # Assign
        dists = torch.cdist(data, centers)
        assignments = dists.argmin(dim=1)

        # Update
        for k in range(num_clusters):
            mask = assignments == k
            if mask.any():
                centers[k] = data[mask].mean(dim=0)

    return centers, assignments
