import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import gc
import math

# ==============================
# 1. CONFIGURATION
# ==============================
BASE_MODEL_ID = "Qwen/Qwen2.5-7B"
train_domains = [
    ("adapters/safe_financial_deividasm_financial_instruction_aq22", "adapters/unsafe_financial_deividasm_financial_instruction_aq22"),
    ("adapters/safe_legal_dzunggg_legal_qa_v1", "adapters/unsafe_legal_dzunggg_legal_qa_v1")
]

print(train_domains)

# ==============================
# 2. DIFFUSION MODEL
# ==============================

class SinusoidalTimestepEmbedding(nn.Module):
    """Maps integer timestep t → continuous embedding via sin/cos encoding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class ResidualBlock(nn.Module):
    """
    A single denoising block.
    Conditions on timestep embedding to let the network behave differently
    at each noise level — crucial for diffusion models.
    """
    def __init__(self, dim, time_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ff1   = nn.Linear(dim, dim * 2)
        self.act   = nn.GELU()
        self.ff2   = nn.Linear(dim * 2, dim)
        self.norm2 = nn.LayerNorm(dim)
        # Projects timestep embedding to match channel dimension
        self.time_proj = nn.Linear(time_dim, dim * 2)

    def forward(self, x, t_emb):
        # Conditioning via AdaLN-style scale/shift
        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale) + shift
        h = self.ff2(self.act(self.ff1(h)))
        return x + self.norm2(h)   # residual


class WeightDiffusionModel(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM) for adapter weight vectors.

    Forward process: adds Gaussian noise to an unsafe weight vector over T steps.
    Reverse process: learned denoiser predicts the noise at each step, guided
    by the unsafe weights as a conditioning signal (classifier-free diffusion).

    Architecture: U-Net-inspired but 1-D — the "U" shape is encoded/decoded
    through a bottleneck, with skip connections bridging encoder and decoder.

    Args:
        dim        : Flat weight vector length (depends on LoRA rank/shape).
        hidden_dim : Width of internal representations (default 512).
        n_blocks   : Depth of encoder AND decoder (total = 2 × n_blocks layers).
        T          : Total diffusion timesteps (more → smoother, slower sampling).
    """
    def __init__(self, dim: int, hidden_dim: int = 512, n_blocks: int = 4, T: int = 1000):
        super().__init__()
        self.T = T
        time_dim = hidden_dim

        # ---------- Timestep encoder ----------
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ---------- Input projection ----------
        # Concatenates noisy target + unsafe conditioning signal
        self.input_proj = nn.Linear(dim * 2, hidden_dim)

        # ---------- Encoder (down-path) ----------
        self.encoder = nn.ModuleList([
            ResidualBlock(hidden_dim, time_dim) for _ in range(n_blocks)
        ])
        self.down_proj = nn.Linear(hidden_dim, hidden_dim // 2)

        # ---------- Bottleneck ----------
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        # ---------- Decoder (up-path with skip connections) ----------
        self.up_proj = nn.Linear(hidden_dim // 2, hidden_dim)
        self.decoder = nn.ModuleList([
            ResidualBlock(hidden_dim, time_dim) for _ in range(n_blocks)
        ])

        # ---------- Output projection ----------
        # Predicts the noise ε added at step t
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim),
        )

        # ---------- Noise schedule (cosine, smoother than linear) ----------
        betas = self._cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",       torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    @staticmethod
    def _cosine_beta_schedule(T: int, s: float = 0.008):
        """Cosine schedule from 'Improved DDPM' (Nichol & Dhariwal, 2021)."""
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        f = f / f[0]
        betas = torch.clamp(1 - f[1:] / f[:-1], min=1e-6, max=0.999)
        return betas.float()

    def q_sample(self, x_safe: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        Forward diffusion: given clean safe weights x_safe, compute noisy version at step t.
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        """
        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_1a = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_a * x_safe + sqrt_1a * noise

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, x_unsafe_cond: torch.Tensor):
        """
        Predicts the noise ε from (noisy safe weights, timestep, unsafe conditioning).
        Concat-conditioning is simple and effective for weight-space guidance.
        """
        t_emb = self.time_mlp(t)
        h = self.input_proj(torch.cat([x_noisy, x_unsafe_cond], dim=-1))

        # Encode
        enc_skip = h
        for block in self.encoder:
            h = block(h, t_emb)
        h = self.down_proj(h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decode with skip
        h = self.up_proj(h)
        h = h + enc_skip  # skip connection from encoder input
        for block in self.decoder:
            h = block(h, t_emb)

        return self.output_proj(h)

    @torch.no_grad()
    def sample(self, x_unsafe_cond: torch.Tensor, n_steps: int = 200):
        """
        DDIM-style deterministic sampling (Fewer steps than full DDPM,
        same quality — fast inference for weight translation).

        Args:
            x_unsafe_cond : (batch, dim) — the unsafe adapter weights to translate.
            n_steps        : inference steps (50-200 is a good range).
        Returns:
            x_safe_pred    : (batch, dim) — predicted safe adapter weights.
        """
        device = x_unsafe_cond.device
        batch  = x_unsafe_cond.shape[0]
        dim    = x_unsafe_cond.shape[1]

        # Start from pure noise, conditioned on the unsafe weights
        x = torch.randn(batch, dim, device=device)

        # Evenly spaced timestep subset (DDIM)
        timesteps = torch.linspace(self.T - 1, 0, n_steps, dtype=torch.long, device=device)

        for i, t_val in enumerate(timesteps):
            t_batch = torch.full((batch,), t_val, device=device, dtype=torch.long)

            # Predict noise
            eps_pred = self(x, t_batch, x_unsafe_cond)

            # DDIM update step
            alpha_t  = self.alphas_cumprod[t_val]
            x0_pred  = (x - (1 - alpha_t).sqrt() * eps_pred) / alpha_t.sqrt()
            x0_pred  = x0_pred.clamp(-10, 10)

            if i < n_steps - 1:
                alpha_prev = self.alphas_cumprod[timesteps[i + 1]]
                x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * eps_pred
            else:
                x = x0_pred

        return x


# ==============================
# 3. DATA COLLECTION (GROUPED BY SHAPE)
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
    # 4. TRAINING LOOP (ONE PER SHAPE)
    # ==============================
    translators = {}

    for shape in X_groups.keys():
        X_train = torch.stack(X_groups[shape])
        Y_train = torch.stack(Y_groups[shape])

        dim = X_train.shape[1]
        print(f"\n--- Training Diffusion Translator for shape {shape} (Dim: {dim}) ---")

        model     = WeightDiffusionModel(dim=dim, hidden_dim=512, n_blocks=4, T=1000).cuda()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        criterion = nn.MSELoss()

        dataset = TensorDataset(X_train.cuda(), Y_train.cuda())
        loader  = DataLoader(dataset, batch_size=4, shuffle=True)

        model.train()
        for epoch in range(200):
            epoch_loss = 0.0

            for x_unsafe, x_safe in loader:
                optimizer.zero_grad()

                # Sample random timesteps for each item in the batch
                t = torch.randint(0, model.T, (x_safe.shape[0],), device=x_safe.device)

                # Forward diffusion: add noise to the safe weights
                noise    = torch.randn_like(x_safe)
                x_noisy  = model.q_sample(x_safe, t, noise)

                # Predict the noise (ε-prediction, standard DDPM objective)
                eps_pred = model(x_noisy, t, x_unsafe)
                loss     = criterion(eps_pred, noise)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            if epoch % 50 == 0:
                print(f"  Epoch {epoch:3d} | Loss: {epoch_loss / len(loader):.6f}")

        translators[str(shape)] = model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return translators


# ==============================
# 5. INFERENCE HELPER
# ==============================

@torch.no_grad()
def translate_weights(translators: dict, shape_key: str, unsafe_weights: torch.Tensor, n_steps: int = 100):
    """
    Convenience wrapper: given a trained translator and an unsafe weight tensor,
    returns the predicted safe weights.

    Args:
        translators   : dict returned by collect_and_train().
        shape_key     : str(shape) — e.g. "torch.Size([64, 64])".
        unsafe_weights: (N, dim) float tensor on CPU.
        n_steps       : DDIM sampling steps.
    """
    model = translators[shape_key].cuda().eval()
    safe_pred = model.sample(unsafe_weights.cuda(), n_steps=n_steps)
    model.cpu()
    return safe_pred.cpu()


# ==============================
# 6. EXECUTE
# ==============================

if __name__ == "__main__":
    os.makedirs("translator_models", exist_ok=True)
    all_translators = collect_and_train()
    torch.save(all_translators, "translator_models/cyber_multi_shape_translators_Diffusion_Qwen2.5_7B.pth")
    print("\nSuccess: Saved diffusion-based safety translators.")