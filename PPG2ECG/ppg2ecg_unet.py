"""
Conditional 1D U-Net Denoiser for PPG → ECG Reconstruction
Architecture: DDIM-compatible conditional denoiser (ε_θ)

Inputs:
  - x_t:     noisy ECG at diffusion timestep t, shape (B, 1, 1024)
  - ppg:     conditioning PPG signal,           shape (B, 1, 1024)
  - t:       diffusion timestep,                shape (B,)

Output:
  - predicted noise ε,                          shape (B, 1, 1024)

Architecture details:
  - Scaling factors: [1, 1, 2, 2, 4, 4, 8] (7 stages)
  - 2 residual blocks per stage
  - Global single-head attention at factor-4 stages (both encoder and decoder)
  - 256-dim sinusoidal time embedding projected into every residual block
  - Dropout 0.2 throughout
  - PPG conditioning via channel concatenation at the input
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Classic sinusoidal positional embedding for diffusion timesteps.
    Produces a (B, dim) embedding from integer timesteps.
    """
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer timesteps
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )                                          # (half,)
        args = t[:, None].float() * freqs[None]   # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return emb


class TimeProjection(nn.Module):
    """Projects the 256-dim time embedding → channel dimension for injection."""
    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.proj(t_emb)  # (B, out_dim)


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    """
    1D residual block with time embedding injection and dropout.
    
    Structure:
      GroupNorm → SiLU → Conv1d
        + time projection (added after first conv)
      GroupNorm → SiLU → Dropout → Conv1d
      + residual projection if in_ch != out_ch
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int = 256, dropout: float = 0.2):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups=min(8, in_ch), num_channels=in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = TimeProjection(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        # Residual projection when channel dims differ
        self.res_proj = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x:     (B, in_ch, L)
        # t_emb: (B, time_dim)
        h = self.conv1(F.silu(self.norm1(x)))               # (B, out_ch, L)
        h = h + self.time_proj(t_emb)[:, :, None]           # broadcast over L
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))  # (B, out_ch, L)
        return h + self.res_proj(x)


# ---------------------------------------------------------------------------
# 1D Global Self-Attention
# ---------------------------------------------------------------------------

class GlobalSelfAttention1D(nn.Module):
    """
    Single-head global self-attention over the sequence dimension.
    Applied at the bottleneck (after factor-4 downsampling).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=1,
            batch_first=True,
            dropout=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        B, C, L = x.shape
        h = self.norm(x)                          # (B, C, L)
        h = h.permute(0, 2, 1)                    # (B, L, C) for MHA
        h, _ = self.attn(h, h, h)                 # (B, L, C)
        h = h.permute(0, 2, 1)                    # (B, C, L)
        return x + h                              # residual


# ---------------------------------------------------------------------------
# Down / Up sampling
# ---------------------------------------------------------------------------

class Downsample1D(nn.Module):
    """Strided conv downsampling by factor `stride`."""
    def __init__(self, channels: int, stride: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """Transposed conv upsampling by factor `stride`."""
    def __init__(self, channels: int, stride: int = 4):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full Conditional 1D U-Net
# ---------------------------------------------------------------------------

class ConditionalUNet1D(nn.Module):
    """
    Conditional 1D U-Net denoiser with proper multi-stage architecture.
    
    Scaling factors: [1, 1, 2, 2, 4, 4, 8] (7 stages)
    - Stages with factor 1: base_channels
    - Stages with factor 2: base_channels * 2
    - Stages with factor 4: base_channels * 4 (with attention at both stages)
    - Stage with factor 8: base_channels * 8 (bottleneck)
    
    Each stage has 2 residual blocks. Attention applied at factor-4 stages only.

    Args:
        signal_length:  Length of input signals (default 1024)
        base_channels:  Channel width at the first conv (default 64)
        time_dim:       Sinusoidal time embedding dimension (default 256)
        dropout:        Dropout ratio (default 0.2)
        n_res_blocks:   Residual blocks per stage (default 2)
    """
    def __init__(
        self,
        signal_length: int = 1024,
        base_channels: int = 64,
        time_dim: int = 256,
        dropout: float = 0.2,
        n_res_blocks: int = 2,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.time_dim = time_dim
        self.n_res_blocks = n_res_blocks
        
        # Define scaling factors and corresponding channels
        # [1, 1, 2, 2, 4, 4, 8]
        scaling_factors = [1, 1, 2, 2, 4, 4, 8]
        channels = [
            base_channels,           # factor 1
            base_channels,           # factor 1
            base_channels * 2,       # factor 2
            base_channels * 2,       # factor 2
            base_channels * 4,       # factor 4 (with attention)
            base_channels * 4,       # factor 4 (with attention)
            base_channels * 8,       # factor 8 (bottleneck)
        ]
        
        # ------------------------------------------------------------------
        # Time embedding
        # ------------------------------------------------------------------
        self.time_emb = SinusoidalTimeEmbedding(dim=time_dim)

        # ------------------------------------------------------------------
        # Input projection: PPG (conditioning) concatenated with x_t → 2 channels
        # ------------------------------------------------------------------
        self.input_proj = nn.Conv1d(2, base_channels, kernel_size=3, padding=1)

        # ------------------------------------------------------------------
        # Encoder: stages with downsampling
        # ------------------------------------------------------------------
        self.enc_stages = nn.ModuleList()
        self.enc_downs = nn.ModuleList()
        
        current_channels = base_channels
        for stage_idx, (factor, out_ch) in enumerate(zip(scaling_factors[:-1], channels[:-1])):
            # Residual blocks for this stage
            blocks = nn.ModuleList([
                ResidualBlock1D(current_channels if i == 0 else out_ch, out_ch, time_dim, dropout)
                for i in range(n_res_blocks)
            ])
            self.enc_stages.append(blocks)
            
            # Attention at factor-4 stages
            if factor == 4:
                self.enc_stages[-1].append(GlobalSelfAttention1D(out_ch))
            
            # Downsampling by this factor
            self.enc_downs.append(Downsample1D(out_ch, stride=factor))
            
            current_channels = out_ch

        # ------------------------------------------------------------------
        # Bottleneck: deepest stage (factor 8)
        # ------------------------------------------------------------------
        bottleneck_ch = channels[-1]
        self.bottleneck_blocks = nn.ModuleList([
            ResidualBlock1D(current_channels if i == 0 else bottleneck_ch, bottleneck_ch, time_dim, dropout)
            for i in range(n_res_blocks)
        ])
        
        # ------------------------------------------------------------------
        # Decoder: stages with upsampling (reverse order)
        # Process in reverse: [8, 4, 4, 2, 2, 1, 1]
        # ------------------------------------------------------------------
        self.dec_ups = nn.ModuleList()
        self.dec_skip_projs = nn.ModuleList()  # Project concatenated skip connections
        self.dec_stages = nn.ModuleList()
        
        current_channels = bottleneck_ch
        for stage_idx in range(len(scaling_factors) - 2, -1, -1):
            factor = scaling_factors[stage_idx]
            out_ch = channels[stage_idx]
            skip_ch = channels[stage_idx]
            
            # Upsampling by this factor
            self.dec_ups.append(Upsample1D(current_channels, stride=factor))
            
            # After upsampling, we concatenate skip connection from encoder
            # We need to project the concatenated (upsampled + skip) back to out_ch
            concat_ch = current_channels + skip_ch
            self.dec_skip_projs.append(nn.Conv1d(concat_ch, out_ch, kernel_size=1))
            
            # Residual blocks for this stage (input after projection is out_ch)
            blocks = nn.ModuleList([
                ResidualBlock1D(out_ch, out_ch, time_dim, dropout)
                for _ in range(n_res_blocks)
            ])
            self.dec_stages.append(blocks)
            
            # Attention at factor-4 stages
            if factor == 4:
                self.dec_stages[-1].append(GlobalSelfAttention1D(out_ch))
            
            current_channels = out_ch

        # ------------------------------------------------------------------
        # Output projection → predicted noise (1 channel ECG)
        # ------------------------------------------------------------------
        self.output_proj = nn.Sequential(
            nn.GroupNorm(num_groups=min(8, current_channels), num_channels=current_channels),
            nn.SiLU(),
            nn.Conv1d(current_channels, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,   # (B, 1, L) noisy ECG at timestep t
        ppg: torch.Tensor,    # (B, 1, L) conditioning PPG signal
        t: torch.Tensor,      # (B,)      integer diffusion timestep
    ) -> torch.Tensor:
        """Returns predicted noise ε, shape (B, 1, L)."""

        # -- Time embedding --------------------------------------------------
        t_emb = self.time_emb(t)                   # (B, 256)

        # -- Condition by concatenation --------------------------------------
        h = torch.cat([x_t, ppg], dim=1)           # (B, 2, L)
        h = self.input_proj(h)                      # (B, base_channels, L)

        # -- Encoder ---------------------------------------------------------
        enc_skips = []
        
        for stage_idx, (down_layer, enc_blocks) in enumerate(zip(self.enc_downs, self.enc_stages)):
            # Apply residual blocks and attention
            for block in enc_blocks:
                if isinstance(block, GlobalSelfAttention1D):
                    h = block(h)
                else:
                    h = block(h, t_emb)
            
            # Store skip connection before downsampling
            enc_skips.append(h)
            
            # Downsample
            h = down_layer(h)

        # -- Bottleneck ------------------------------------------------------
        for block in self.bottleneck_blocks:
            h = block(h, t_emb)

        # -- Decoder ---------------------------------------------------------
        for stage_idx, (up_layer, skip_proj, dec_blocks) in enumerate(
            zip(self.dec_ups, self.dec_skip_projs, self.dec_stages)
        ):
            # Upsample
            h = up_layer(h)
            
            # Concatenate skip connection
            skip = enc_skips[-(stage_idx + 1)]
            h = torch.cat([h, skip], dim=1)
            
            # Project concatenated features back to stage output channels
            h = skip_proj(h)
            
            # Apply residual blocks and attention
            for block in dec_blocks:
                if isinstance(block, GlobalSelfAttention1D):
                    h = block(h)
                else:
                    h = block(h, t_emb)

        # -- Output ----------------------------------------------------------
        return self.output_proj(h)                  # (B, 1, L)


# ---------------------------------------------------------------------------
# DDIM inference utility
# ---------------------------------------------------------------------------

class DDIMSampler:
    """
    Minimal DDIM sampler for inference (no training logic).
    
    Usage:
        sampler = DDIMSampler(model, n_timesteps=1000)
        ecg = sampler.sample(ppg, n_steps=50)  # fast sampling with 50 steps
    """
    def __init__(self, model: ConditionalUNet1D, n_timesteps: int = 1000, beta_start: float = 1e-6, beta_end: float = 1e-2):
        self.model = model
        self.T = n_timesteps

        betas = torch.linspace(beta_start, beta_end, n_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_t

    @torch.no_grad()
    def sample(
        self,
        ppg: torch.Tensor,      # (B, 1, L)
        n_steps: int = 50,      # DDIM steps (can be << T)
        device: str = "cpu",
    ) -> torch.Tensor:
        """Reconstruct ECG from PPG using DDIM reverse process."""
        self.model.eval()
        B, _, L = ppg.shape
        ppg = ppg.to(device)
        alphas_cumprod = self.alphas_cumprod.to(device)

        # Subsample timestep schedule
        step_indices = torch.linspace(0, self.T - 1, n_steps + 1, dtype=torch.long)
        timesteps = step_indices.flip(0)  # go from T → 0

        # Start from pure noise
        x = torch.randn(B, 1, L, device=device)

        for i in range(len(timesteps) - 1):
            t_now = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = t_now.expand(B).to(device)
            alpha_now = alphas_cumprod[t_now]
            alpha_next = alphas_cumprod[t_next]

            # Predict noise
            eps = self.model(x, ppg, t_batch)

            # DDIM update
            x0_pred = (x - (1 - alpha_now).sqrt() * eps) / alpha_now.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)
            x = alpha_next.sqrt() * x0_pred + (1 - alpha_next).sqrt() * eps

        return x  # reconstructed ECG, shape (B, 1, L)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    model = ConditionalUNet1D(
        signal_length=1024,
        base_channels=64,
        time_dim=256,
        dropout=0.2,
        n_res_blocks=2,
    ).to(device)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"Model size (fp32): {size_mb:.2f} MB")

    # Forward pass
    B = 4
    x_t = torch.randn(B, 1, 1024).to(device)
    ppg = torch.randn(B, 1, 1024).to(device)
    t   = torch.randint(0, 1000, (B,)).to(device)

    eps_pred = model(x_t, ppg, t)
    print(f"Input shape:  {x_t.shape}")
    print(f"Output shape: {eps_pred.shape}")
    assert eps_pred.shape == (B, 1, 1024), "Shape mismatch!"
    print("Forward pass OK")

    # DDIM sampling
    sampler = DDIMSampler(model)
    ecg_recon = sampler.sample(ppg, n_steps=50, device=device)
    print(f"Reconstructed ECG shape: {ecg_recon.shape}")
    print("DDIM sampling OK")
