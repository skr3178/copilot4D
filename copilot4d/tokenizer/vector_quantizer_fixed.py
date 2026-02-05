"""Fixed Vector Quantizer to address codebook collapse.

Key fixes:
1. EMA-based codebook update (more stable than gradient-based)
2. Correct loss coefficient ordering
3. Periodic (not per-step) dead code detection
4. Better initialization and gradient scaling
5. Comprehensive metrics for monitoring codebook health
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import math


class VectorQuantizerFixed(nn.Module):
    """Fixed Vector Quantization with EMA codebook and better stability.
    
    Key changes from original:
    - Uses EMA for codebook updates instead of pure gradient
    - Fixed loss coefficients (lambda_1 for codebook, lambda_2 for commitment)
    - Periodic dead code re-init (every N steps, not every step)
    - Better initialization with unit norm codebook
    - Comprehensive metrics: perplexity, entropy, usage %, dead codes, loss components
    """

    def __init__(
        self,
        dim: int = 256,
        codebook_size: int = 1024,
        codebook_dim: int = 1024,
        commitment_cost: float = 1.0,    # lambda_2: encoder commitment
        codebook_cost: float = 0.25,      # lambda_1: codebook update
        decay: float = 0.99,              # EMA decay for codebook
        epsilon: float = 1e-5,
        kmeans_iters: int = 10,
        dead_threshold: int = 256,
        dead_percentage: float = 0.03,
        min_iterations: int = 200,
        reinit_every: int = 100,          # Only check dead codes every N steps
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost  # lambda_2 = 1.0 (encoder)
        self.codebook_cost = codebook_cost      # lambda_1 = 0.25 (codebook)
        self.decay = decay
        self.epsilon = epsilon
        self.kmeans_iters = kmeans_iters
        self.dead_threshold = dead_threshold
        self.dead_percentage = dead_percentage
        self.min_iterations = min_iterations
        self.reinit_every = reinit_every

        # Pre-VQ projection: encoder output -> codebook space
        self.pre_norm = nn.LayerNorm(dim)
        self.pre_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, codebook_dim),
        )

        # Post-VQ projection: codebook -> encoder output space
        self.post_proj = nn.Linear(codebook_dim, dim)

        # Codebook with proper initialization
        self.register_buffer("embed", torch.randn(codebook_size, codebook_dim))
        # Normalize to unit sphere for better stability
        self.embed.data = F.normalize(self.embed.data, dim=1) * (codebook_dim ** 0.5)
        
        # EMA cluster size and embeddings
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", self.embed.data.clone())
        
        # Usage tracking
        self.register_buffer("usage_count", torch.zeros(codebook_size, dtype=torch.long))
        self.register_buffer("iteration_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("initted", torch.tensor(False))

        # Memory bank for dead code restart
        memory_bank_size = 10 * codebook_size
        self.register_buffer("memory_bank", torch.zeros(memory_bank_size, codebook_dim))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _update_memory_bank(self, data: torch.Tensor):
        """Store encoder outputs in memory bank (circular buffer)."""
        batch_size = data.shape[0]
        ptr = int(self.bank_ptr)
        bank_size = self.memory_bank.shape[0]
        
        if ptr + batch_size > bank_size:
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
        self.embed_avg.data.copy_(embed)
        self.initted.fill_(True)

    @torch.no_grad()
    def _update_codebook_ema(self, flat: torch.Tensor, indices: torch.Tensor):
        """Update codebook using EMA.
        
        This is more stable than gradient-based updates because:
        1. Codebook moves gradually toward encoder outputs
        2. Doesn't suffer from gradient variance
        3. Maintains better codebook diversity
        """
        # One-hot encoding of indices
        encodings = F.one_hot(indices, self.codebook_size).float()  # (N, K)
        
        # Update cluster sizes with EMA
        self.cluster_size = self.cluster_size * self.decay + \
                           (1 - self.decay) * encodings.sum(dim=0)  # (K,)
        
        # Update embeddings with EMA
        embed_sum = flat.t() @ encodings  # (D, K)
        self.embed_avg = self.embed_avg * self.decay + (1 - self.decay) * embed_sum.t()
        
        # Laplace smoothing to avoid empty clusters
        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
        )
        
        # Normalize embeddings
        self.embed = self.embed_avg / cluster_size.unsqueeze(1)

    @torch.no_grad()
    def _check_and_reinit_dead_codes(self):
        """Check for dead codes and re-initialize entire codebook with K-Means.
        
        Following the paper (Section 4.1):
        - A code is "dead" if not used for 256 iterations
        - If >3% of codebook is dead, run K-Means on memory bank
        - Re-initialize the ENTIRE codebook with K-Means centroids
        - Each codebook must go through at least 200 iterations before re-init
        """
        self.iteration_count += 1
        
        # Only check periodically, not every step
        if self.iteration_count % self.reinit_every != 0:
            return
        
        if self.iteration_count < self.min_iterations:
            return
        
        # Find dead codes (not used for dead_threshold iterations)
        dead_codes = self.usage_count >= self.dead_threshold
        num_dead = dead_codes.sum().item()
        dead_ratio = num_dead / self.codebook_size
        
        if dead_ratio > self.dead_percentage:
            print(f"[VQ] Codebook collapse detected: {num_dead} dead codes ({100*dead_ratio:.1f}%)")
            print(f"[VQ] Re-initializing ENTIRE codebook with K-Means on memory bank...")
            
            bank_data = self.memory_bank[self.memory_bank.abs().sum(dim=1) > 0]
            if bank_data.shape[0] >= self.codebook_size:
                # Run K-Means on memory bank to get new codebook centroids
                # Paper: "run K-Means clustering on the memory bank to re-initialize 
                #         the entire codebook"
                new_embed, _ = _kmeans(bank_data, self.codebook_size, self.kmeans_iters)
                
                # Replace ENTIRE codebook with K-Means centroids
                self.embed.data.copy_(new_embed)
                self.embed_avg.data.copy_(new_embed)
                
                # Reset cluster sizes uniformly
                self.cluster_size.fill_(1.0)
                
                # Reset usage counters
                self.usage_count.zero_()
                
                print(f"[VQ] Codebook re-initialized with K-Means centroids from {bank_data.shape[0]} samples")

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, dim) encoder output

        Returns:
            quantized: (B, N, dim)
            indices: (B, N)
            vq_loss: scalar
            metrics: dict with perplexity and usage info
        """
        B, N, D = x.shape

        # Pre-projection (all in FP32)
        x_normed = self.pre_norm(x.float())
        z_e = self.pre_proj(x_normed)  # (B, N, codebook_dim)

        flat = z_e.reshape(-1, self.codebook_dim)  # (B*N, codebook_dim)

        # K-Means init on first batch
        if self.training and not self.initted:
            self._init_codebook_kmeans(flat)

        # Update memory bank
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
            self.usage_count += 1
            used_codes = torch.unique(indices)
            self.usage_count[used_codes] = 0
            
            # Update codebook with EMA
            self._update_codebook_ema(flat, indices)
            
            # Periodic dead code check
            self._check_and_reinit_dead_codes()

        # VQ Loss (paper: L_vq = 0.25 * ||sg[E(o)] - z_hat||^2 + 1.0 * ||sg[z_hat] - E(o)||^2)
        # Where:
        #   - First term (codebook_cost=0.25): codebook moves toward encoder outputs
        #   - Second term (commitment_cost=1.0): encoder commits to codebook
        
        # Codebook loss: ||sg(z_e) - z_q||^2
        codebook_loss = self.codebook_cost * F.mse_loss(flat.detach(), z_q)
        
        # Commitment loss: ||z_e - sg(z_q)||^2  
        commitment_loss = self.commitment_cost * F.mse_loss(flat, z_q.detach())
        
        vq_loss = codebook_loss + commitment_loss

        # Compute comprehensive metrics for monitoring codebook health
        with torch.no_grad():
            metrics = self._compute_metrics(indices, codebook_loss, commitment_loss)

        # Straight-through estimator
        z_q_st = flat + (z_q - flat).detach()
        z_q_st = z_q_st.view(B, N, self.codebook_dim)

        # Post-projection back to dim
        quantized = self.post_proj(z_q_st)
        indices = indices.view(B, N)

        return quantized, indices, vq_loss, metrics

    def _compute_metrics(
        self, 
        indices: torch.Tensor, 
        codebook_loss: torch.Tensor,
        commitment_loss: torch.Tensor
    ) -> dict:
        """Compute comprehensive codebook health metrics.
        
        Args:
            indices: (N,) quantized indices
            codebook_loss: scalar codebook loss component
            commitment_loss: scalar commitment loss component
            
        Returns:
            dict with codebook health metrics
        """
        N = indices.shape[0]
        encodings = F.one_hot(indices, self.codebook_size).float()  # (N, K)
        
        # Usage counts per code
        usage_counts = encodings.sum(dim=0)  # (K,)
        
        # Active codes (used at least once)
        active_codes = (usage_counts > 0).sum().item()
        active_pct = 100.0 * active_codes / self.codebook_size
        
        # Perplexity: exp(entropy) - how uniformly codes are used
        # High perplexity (~codebook_size) = good utilization
        # Low perplexity (~1) = collapse (few codes used)
        avg_probs = usage_counts / N
        entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
        perplexity = torch.exp(entropy)
        
        # Codebook entropy (bits per code)
        # Max entropy = log2(K) when uniform
        max_entropy = math.log2(self.codebook_size)
        normalized_entropy = entropy.item() / max_entropy if max_entropy > 0 else 0
        
        # Cluster size statistics (detect imbalance)
        nonzero_usage = usage_counts[usage_counts > 0]
        if len(nonzero_usage) > 0:
            max_cluster = nonzero_usage.max().item()
            mean_cluster = nonzero_usage.mean().item()
            min_cluster = nonzero_usage.min().item()
            std_cluster = nonzero_usage.std().item()
        else:
            max_cluster = mean_cluster = min_cluster = std_cluster = 0
        
        # Detect potential dead codes (not used recently)
        # Track which codes haven't been used in this batch
        dead_codes = self.codebook_size - active_codes
        
        # Loss component ratio (helps diagnose training balance)
        # If commitment >> codebook, encoder is learning too fast
        loss_ratio = (commitment_loss / (codebook_loss + 1e-8)).item()
        
        # Code usage distribution metrics
        # Gini coefficient for inequality (0 = perfect equality, 1 = max inequality)
        if active_codes > 0:
            sorted_usage, _ = torch.sort(usage_counts)
            sorted_usage = sorted_usage[-active_codes:]  # Only active codes
            n = active_codes
            index = torch.arange(1, n + 1, device=sorted_usage.device, dtype=torch.float32)
            gini = (2 * (index * sorted_usage).sum()) / (n * sorted_usage.sum()) - (n + 1) / n
            gini = gini.item()
        else:
            gini = 1.0  # Max inequality if no codes used
        
        metrics = {
            # Core codebook health metrics
            "vq_perplexity": perplexity.item(),
            "vq_entropy": entropy.item(),
            "vq_entropy_norm": normalized_entropy,
            "vq_active_codes": active_codes,
            "vq_active_pct": active_pct,
            "vq_dead_codes": dead_codes,
            "vq_dead_pct": 100.0 * dead_codes / self.codebook_size,
            
            # Cluster statistics
            "vq_cluster_max": max_cluster,
            "vq_cluster_mean": mean_cluster,
            "vq_cluster_min": min_cluster,
            "vq_cluster_std": std_cluster,
            "vq_gini": gini,  # Usage inequality
            
            # Loss components
            "vq_loss_codebook": codebook_loss.item(),
            "vq_loss_commitment": commitment_loss.item(),
            "vq_loss_ratio": loss_ratio,  # commitment / codebook
            
            # Reference values for healthy training
            "vq_max_entropy": max_entropy,
            "vq_codebook_size": self.codebook_size,
        }
        
        return metrics

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
    """K-Means clustering for codebook initialization."""
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
