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
    ("adapters/safe_legal_dzunggg_legal_qa_v1","adapters/unsafe_legal_dzunggg_legal_qa_v1")
]

print(train_domains)

# ==============================
# 2. FLOW MATCHING MODEL
# ==============================
#
# Conditional Flow Matching (CFM) — Lipman et al., 2022.
#
# Core idea:
#   Instead of a noisy diffusion process, we define a straight-line interpolation
#   between two distributions:
#
#       x(t) = (1 - t) * x_source  +  t * x_target        t ∈ [0, 1]
#
#   The constant velocity along this path is:
#
#       v* = x_target - x_source
#
#   The network learns to predict this velocity v* from (x(t), t, conditioning).
#   At inference we integrate v from t=0 to t=1 with a simple ODE solver.
#
# Why this beats DDPM for small datasets:
#   - Straight-line paths are the "shortest" paths between distributions,
#     so the velocity field is simpler to learn.
#   - No noise schedule to tune.
#   - Only one hyperparameter that matters: the number of ODE steps at inference.
#   - Training loss is always MSE on the velocity (no SNR weighting needed).
#
# Conditioning strategy used here:
#   Source = unsafe adapter weights (x0)
#   Target = safe adapter weights   (x1)
#   The network gets x(t) AND x0 concatenated, so it always knows "where we started".


class SinusoidalTimeEmbedding(nn.Module):
    """Scalar t ∈ [0,1] → fixed-frequency embedding → projected to `dim`."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float in [0, 1]
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args   = t[:, None] * freqs[None]
        emb    = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


class VelocityBlock(nn.Module):
    """
    One residual block inside the velocity network.
    AdaLN conditioning: time embedding modulates LayerNorm output via scale/shift.
    """
    def __init__(self, dim: int, time_dim: int, expansion: int = 2):
        super().__init__()
        inner = dim * expansion
        self.norm     = nn.LayerNorm(dim)
        self.ff1      = nn.Linear(dim, inner)
        self.act      = nn.GELU()
        self.ff2      = nn.Linear(inner, dim)
        self.time_mod = nn.Linear(time_dim, dim * 2)   # → scale, shift

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.time_mod(t_emb).chunk(2, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        h = self.ff2(self.act(self.ff1(h)))
        return x + h


class WeightFlowMatchingModel(nn.Module):
    """
    Velocity network for Conditional Flow Matching.

    Input  : x(t)  (interpolated weights at time t) ++ x0 (unsafe conditioning)
    Output : v(x,t) — predicted velocity (should approximate x1 - x0)

    At training:  loss = MSE(v_pred,  x_safe - x_unsafe)
    At inference: Euler / RK4 integration of v from t=0 to t=1,
                  starting from x_unsafe (or from Gaussian noise — see sample()).

    Args:
        dim        : Flat weight vector length.
        hidden_dim : Internal width.
        n_blocks   : Number of VelocityBlocks (depth).
    """
    def __init__(self, dim: int, hidden_dim: int = 512, n_blocks: int = 6):
        super().__init__()
        self.dim        = dim
        time_dim        = hidden_dim

        # Timestep embedding (t is a scalar in [0,1] here, not an integer)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        # Project concatenated (x_t, x0_cond) → hidden
        self.input_proj = nn.Linear(dim * 2, hidden_dim)

        # Deep residual velocity network
        self.blocks = nn.ModuleList([
            VelocityBlock(hidden_dim, time_dim) for _ in range(n_blocks)
        ])

        # Output: velocity in the same space as the weights
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim),
        )

        # Zero-init the final linear so velocity starts near 0 — stabilises early training
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, x0_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t     : (B, dim)  interpolated sample at time t
            t       : (B,)      float in [0, 1]
            x0_cond : (B, dim)  unsafe adapter weights (conditioning)
        Returns:
            v       : (B, dim)  predicted velocity
        """
        t_emb = self.time_embed(t)
        h     = self.input_proj(torch.cat([x_t, x0_cond], dim=-1))
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)

    @torch.no_grad()
    def sample(self, x_unsafe: torch.Tensor, n_steps: int = 100, solver: str = "rk4") -> torch.Tensor:
        """
        Integrate the learned velocity field from t=0 to t=1.

        Two solvers:
          'euler' — first-order, fast, good for n_steps ≥ 100.
          'rk4'   — fourth-order Runge-Kutta, accurate at n_steps = 20-50.

        We start at x_unsafe (the source distribution), not pure noise.
        This is "source-conditioned" flow matching — the ODE transports the
        unsafe distribution directly to the safe distribution.

        Args:
            x_unsafe : (B, dim) unsafe weights on the correct device.
            n_steps  : ODE discretisation steps.
            solver   : 'euler' or 'rk4'.
        Returns:
            x_safe   : (B, dim) predicted safe weights.
        """
        device = x_unsafe.device
        x      = x_unsafe.clone()
        dt     = 1.0 / n_steps
        ts     = torch.linspace(0.0, 1.0 - dt, n_steps, device=device)

        for t_val in ts:
            t_batch = torch.full((x.shape[0],), t_val, device=device)

            if solver == "euler":
                v = self(x, t_batch, x_unsafe)
                x = x + v * dt

            elif solver == "rk4":
                # Standard RK4 — each sub-step re-evaluates the velocity field
                def v_fn(x_, t_):
                    tb = torch.full((x_.shape[0],), t_, device=device)
                    return self(x_, tb, x_unsafe)

                k1 = v_fn(x,                t_val)
                k2 = v_fn(x + 0.5*dt*k1,   t_val + 0.5*dt)
                k3 = v_fn(x + 0.5*dt*k2,   t_val + 0.5*dt)
                k4 = v_fn(x +     dt*k3,   t_val +     dt)
                x  = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

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
        X_train = torch.stack(X_groups[shape])   # (N, dim)  — unsafe
        Y_train = torch.stack(Y_groups[shape])   # (N, dim)  — safe

        dim = X_train.shape[1]
        print(f"\n--- Training Flow Matching Translator | shape {shape} | dim {dim} ---")

        model     = WeightFlowMatchingModel(dim=dim, hidden_dim=512, n_blocks=6).cuda()
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
        criterion = nn.MSELoss()

        dataset = TensorDataset(X_train.cuda(), Y_train.cuda())
        loader  = DataLoader(dataset, batch_size=4, shuffle=True)

        model.train()
        for epoch in range(300):
            epoch_loss = 0.0

            for x_unsafe, x_safe in loader:
                optimizer.zero_grad()

                # ---- Flow Matching training objective ----
                # 1. Sample random t ∈ [0, 1] uniformly for each item
                t = torch.rand(x_safe.shape[0], device=x_safe.device)

                # 2. Interpolate: x(t) = (1-t)*x_unsafe + t*x_safe
                t_col = t.view(-1, 1)
                x_t   = (1 - t_col) * x_unsafe + t_col * x_safe

                # 3. Target velocity: constant along straight path
                v_target = x_safe - x_unsafe

                # 4. Predict velocity and regress
                v_pred = model(x_t, t, x_unsafe)
                loss   = criterion(v_pred, v_target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            if epoch % 50 == 0:
                avg_loss = epoch_loss / len(loader)
                print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

        translators[str(shape)] = model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return translators


# ==============================
# 5. INFERENCE HELPER
# ==============================

@torch.no_grad()
def translate_weights(translators: dict, shape_key: str,
                      unsafe_weights: torch.Tensor,
                      n_steps: int = 50, solver: str = "rk4") -> torch.Tensor:
    """
    Translate a batch of unsafe adapter weights to safe weights.

    Args:
        translators    : dict returned by collect_and_train().
        shape_key      : str(shape) — e.g. "torch.Size([64, 64])".
        unsafe_weights : (N, dim) float tensor on CPU.
        n_steps        : ODE steps (50 with rk4 ≈ 200 Euler steps in quality).
        solver         : 'euler' or 'rk4'.
    Returns:
        safe_weights   : (N, dim) float tensor on CPU.
    """
    model     = translators[shape_key].cuda().eval()
    safe_pred = model.sample(unsafe_weights.cuda(), n_steps=n_steps, solver=solver)
    model.cpu()
    return safe_pred.cpu()


# ==============================
# 6. EXECUTE
# ==============================

if __name__ == "__main__":
    os.makedirs("translator_models", exist_ok=True)
    all_translators = collect_and_train()
    torch.save(
        all_translators,
        "translator_models/cyber_multi_shape_translators_Flow_Matching_Qwen2.5_7B.pth"
    )
    print("\nSuccess: Saved flow matching safety translators.")