import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import gc

# ==============================
# 1. CONFIGURATION
# ==============================
BASE_MODEL_ID = "Qwen/Qwen2.5-7B"
train_domains = [
    ("adapters/safe_financial_deividasm_financial_instruction_aq22", "adapters/unsafe_financial_deividasm_financial_instruction_aq22"),
    ("adapters/safe_legal_dzunggg_legal_qa_v1",                     "adapters/unsafe_legal_dzunggg_legal_qa_v1")
]

print(train_domains)

# ==============================
# 2. CONDITIONAL VAE
# ==============================
#
# Architecture: Conditional VAE (CVAE) — Sohn et al., 2015.
#
# Why CVAE works well for this task:
#   - The encoder sees BOTH (unsafe, safe) at training time, letting it learn
#     what "transformation residual" needs to be stored in z.
#   - The KL term regularises z toward N(0,I), so at inference (when we only
#     have unsafe weights) sampling z ~ N(0,I) still produces plausible safe weights.
#   - With very few pairs, the KL bottleneck prevents the model from memorising
#     pair-specific transformations — it is forced to generalise.
#   - Posterior collapse mitigation: we use KL annealing (β ramps from 0 → 1
#     over the first half of training) and a controlled latent dimension.
#
# Training:
#   Encoder q(z | x_unsafe, x_safe) → μ, log σ²
#   Decoder p(x_safe | x_unsafe, z) → x̂_safe
#   Loss = Recon(x̂_safe, x_safe) + β * KL(q || N(0,I))
#
# Inference:
#   z ~ N(0, I)
#   x̂_safe = Decoder(x_unsafe, z)


class ResBlock(nn.Module):
    """Lightweight residual MLP block used in both encoder and decoder."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CVAEEncoder(nn.Module):
    """
    Recognition network q(z | x_unsafe, x_safe).

    Sees BOTH inputs at training time to infer what transformation was applied.
    Outputs μ and log σ² of the approximate posterior.

    Args:
        weight_dim  : Flat weight vector length.
        hidden_dim  : Internal width.
        latent_dim  : Dimensionality of z (keep small: 32–128 is enough).
        n_blocks    : Residual depth.
    """
    def __init__(self, weight_dim: int, hidden_dim: int, latent_dim: int, n_blocks: int = 3):
        super().__init__()
        # Concatenate (x_unsafe, x_safe) as input
        self.input_proj = nn.Linear(weight_dim * 2, hidden_dim)
        self.blocks     = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.mu_head    = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Zero-init heads → starts near prior N(0,I), avoids early KL explosion
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -2.0)  # start with small variance

    def forward(self, x_unsafe: torch.Tensor, x_safe: torch.Tensor):
        h = self.input_proj(torch.cat([x_unsafe, x_safe], dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.mu_head(h), self.logvar_head(h)


class CVAEDecoder(nn.Module):
    """
    Generative network p(x_safe | x_unsafe, z).

    At training: z is sampled from the encoder posterior.
    At inference: z is sampled from the prior N(0,I).

    The decoder always has access to x_unsafe as conditioning — this is the
    key difference from a plain VAE and is what lets it produce the *right*
    safe weights for a specific unsafe input.

    Args:
        weight_dim  : Flat weight vector length.
        hidden_dim  : Internal width.
        latent_dim  : Must match encoder.
        n_blocks    : Residual depth.
    """
    def __init__(self, weight_dim: int, hidden_dim: int, latent_dim: int, n_blocks: int = 3):
        super().__init__()
        # Concatenate (x_unsafe, z) as input
        self.input_proj = nn.Linear(weight_dim + latent_dim, hidden_dim)
        self.blocks     = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, weight_dim),
        )

    def forward(self, x_unsafe: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(torch.cat([x_unsafe, z], dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class WeightCVAE(nn.Module):
    """
    Full Conditional VAE for adapter weight translation.

    Args:
        weight_dim  : Flat weight vector length.
        hidden_dim  : Internal width (default 512).
        latent_dim  : z dimensionality (default 64).
        n_blocks    : Depth of encoder and decoder (default 3 each).
    """
    def __init__(self, weight_dim: int, hidden_dim: int = 512,
                 latent_dim: int = 64, n_blocks: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = CVAEEncoder(weight_dim, hidden_dim, latent_dim, n_blocks)
        self.decoder = CVAEDecoder(weight_dim, hidden_dim, latent_dim, n_blocks)

    @staticmethod
    def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        z = μ + ε * σ,   ε ~ N(0,I)
        Differentiable sampling via the reparametrization trick.
        logvar is clamped to [-10, 2] to prevent numerical instability.
        """
        logvar = logvar.clamp(-10.0, 2.0)
        std    = (0.5 * logvar).exp()
        eps    = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_unsafe: torch.Tensor, x_safe: torch.Tensor):
        """
        Training forward pass.
        Returns:
            x_safe_pred : reconstructed safe weights
            mu          : posterior mean
            logvar      : posterior log variance
        """
        mu, logvar   = self.encoder(x_unsafe, x_safe)
        z            = self.reparametrize(mu, logvar)
        x_safe_pred  = self.decoder(x_unsafe, z)
        return x_safe_pred, mu, logvar

    @torch.no_grad()
    def sample(self, x_unsafe: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Inference: draw z from the prior N(0,I) and decode.

        Args:
            x_unsafe  : (B, dim)  unsafe adapter weights.
            n_samples : how many independent safe-weight proposals to average.
                        n_samples=1 is the standard single-pass prediction.
                        n_samples>1 gives a Monte Carlo average — more stable
                        when the posterior has high variance.
        Returns:
            x_safe    : (B, dim) predicted safe weights.
        """
        device = x_unsafe.device
        B, dim = x_unsafe.shape

        if n_samples == 1:
            z = torch.randn(B, self.latent_dim, device=device)
            return self.decoder(x_unsafe, z)

        # Average over multiple z samples to reduce variance
        preds = torch.stack([
            self.decoder(x_unsafe, torch.randn(B, self.latent_dim, device=device))
            for _ in range(n_samples)
        ])  # (n_samples, B, dim)
        return preds.mean(dim=0)


# ==============================
# 3. LOSS FUNCTION
# ==============================

def cvae_loss(x_safe_pred: torch.Tensor, x_safe: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
    """
    ELBO loss: Reconstruction + β * KL divergence.

    Reconstruction: MSE between predicted and true safe weights.
    KL: -0.5 * sum(1 + log σ² - μ² - σ²)   analytically from N(μ,σ²) to N(0,I).
    β:  annealing coefficient ramped from 0 → 1 during training to prevent
        posterior collapse (the encoder collapsing to the prior before the
        decoder learns to use z).

    Args:
        x_safe_pred : (B, dim) decoder output
        x_safe      : (B, dim) ground truth
        mu          : (B, latent_dim) posterior mean
        logvar      : (B, latent_dim) posterior log variance
        beta        : KL weight (anneal from 0 to 1)
    Returns:
        total_loss, recon_loss, kl_loss  — all scalars
    """
    recon = nn.functional.mse_loss(x_safe_pred, x_safe, reduction="mean")
    kl    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return recon + beta * kl, recon, kl


# ==============================
# 4. DATA COLLECTION (GROUPED BY SHAPE)
# ==============================

def collect_and_train():
    from safetensors.torch import load_file

    X_groups = {}
    Y_groups = {}

    print("Step 1: Categorizing weights by shape...")
    for path_safe, path_unsafe in train_domains:
        s_w = load_file(os.path.join(path_safe,   "adapter_model.safetensors"))
        u_w = load_file(os.path.join(path_unsafe, "adapter_model.safetensors"))

        for k in s_w.keys():
            if k in u_w:
                shape = s_w[k].shape
                if shape not in X_groups:
                    X_groups[shape] = []
                    Y_groups[shape] = []
                X_groups[shape].append(u_w[k].flatten().float())
                Y_groups[shape].append(s_w[k].flatten().float())

    # ==============================
    # 5. TRAINING LOOP (ONE PER SHAPE)
    # ==============================
    translators = {}

    for shape in X_groups.keys():
        X_train = torch.stack(X_groups[shape])
        Y_train = torch.stack(Y_groups[shape])

        dim = X_train.shape[1]
        print(f"\n--- Training CVAE Translator | shape {shape} | dim {dim} ---")

        model     = WeightCVAE(weight_dim=dim, hidden_dim=256, latent_dim=32, n_blocks=2).cuda()
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

        # Keep data on CPU, move batches on-the-fly to avoid holding two full
        # weight matrices on GPU simultaneously (the main cause of OOM here)
        dataset = TensorDataset(X_train, Y_train)
        loader  = DataLoader(dataset, batch_size=1, shuffle=True)

        n_epochs     = 300
        warmup_frac  = 0.4   # KL weight ramps from 0 → 1 over first 40% of training

        model.train()
        for epoch in range(n_epochs):
            epoch_loss  = 0.0
            epoch_recon = 0.0
            epoch_kl    = 0.0

            # KL annealing: β = 0 initially so decoder learns to reconstruct first
            beta = min(1.0, epoch / (n_epochs * warmup_frac))

            for x_unsafe, x_safe in loader:
                x_unsafe = x_unsafe.cuda()
                x_safe   = x_safe.cuda()
                optimizer.zero_grad()

                x_safe_pred, mu, logvar = model(x_unsafe, x_safe)
                loss, recon, kl         = cvae_loss(x_safe_pred, x_safe, mu, logvar, beta)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss  += loss.item()
                epoch_recon += recon.item()
                epoch_kl    += kl.item()

            scheduler.step()

            if epoch % 50 == 0:
                n = len(loader)
                print(f"  Epoch {epoch:3d} | β={beta:.2f} | "
                      f"Loss: {epoch_loss/n:.5f}  "
                      f"Recon: {epoch_recon/n:.5f}  "
                      f"KL: {epoch_kl/n:.5f}")

        translators[str(shape)] = model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return translators


# ==============================
# 6. INFERENCE HELPER
# ==============================

@torch.no_grad()
def translate_weights(translators: dict, shape_key: str,
                      unsafe_weights: torch.Tensor,
                      n_samples: int = 10) -> torch.Tensor:
    """
    Translate a batch of unsafe adapter weights to safe weights via the CVAE.

    Args:
        translators    : dict returned by collect_and_train().
        shape_key      : str(shape) — e.g. "torch.Size([64, 64])".
        unsafe_weights : (N, dim) float tensor on CPU.
        n_samples      : number of z samples to average for stable output.
                         10 is a safe default; use 1 for maximum speed.
    Returns:
        safe_weights   : (N, dim) float tensor on CPU.
    """
    model     = translators[shape_key].cuda().eval()
    safe_pred = model.sample(unsafe_weights.cuda(), n_samples=n_samples)
    model.cpu()
    return safe_pred.cpu()


# ==============================
# 7. EXECUTE
# ==============================

if __name__ == "__main__":
    os.makedirs("translator_models", exist_ok=True)
    all_translators = collect_and_train()
    torch.save(
        all_translators,
        "translator_models/cyber_multi_shape_translators_CVAE_Qwen2.5_7B_FL.pth"
    )
    print("\nSuccess: Saved CVAE safety translators.")