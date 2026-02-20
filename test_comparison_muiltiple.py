import torch
import torch.nn.functional as F
import os
from collections import Counter

# ==============================
# 1. CONFIGURATION
# ==============================
DOMAINS = {
    "Medical": ("./final_adapters/medical_base_safe", "./final_adapters/medical_base_unsafe"),
    "Cyber":   ("./final_adapters/cyber_base_safe",   "./final_adapters/cyber_base_unsafe"),
    "Finance": ("./final_adapters/fin_base_safe", "./final_adapters/fin_base_unsafe")
}

def load_raw_weights(path):
    from safetensors.torch import load_file
    sf_path = os.path.join(path, "adapter_model.safetensors")
    bin_path = os.path.join(path, "adapter_model.bin")
    w = load_file(sf_path, device="cpu") if os.path.exists(sf_path) else torch.load(bin_path, map_location="cpu")
    
    # Standardize keys to 'layers.X.submodule'
    clean = {}
    for k, v in w.items():
        clean_k = k.replace("base_model.model.model.", "").replace("base_model.model.", "")
        clean_k = ".".join([p for p in clean_k.split(".") if p not in ["u", "s", "intervention", "safe", "unsafe", "default", "weight"]])
        clean[clean_k] = v.float()
    return clean

# ==============================
# 2. COLLECT TOP DIVERGENT LAYERS
# ==============================
all_domain_rankings = {}

for domain, paths in DOMAINS.items():
    print(f"Analyzing {domain}...")
    s_weights = load_raw_weights(paths[0])
    u_weights = load_raw_weights(paths[1])
    
    layer_sims = []
    for key in s_weights.keys():
        if "lora_B" in key and key in u_weights:
            sim = F.cosine_similarity(s_weights[key].flatten().unsqueeze(0), 
                                     u_weights[key].flatten().unsqueeze(0)).item()
            # ID looks like: layers.15.mlp.down_proj
            layer_sims.append((key, sim))
    
    # Sort by lowest similarity (most divergent)
    layer_sims.sort(key=lambda x: x[1])
    all_domain_rankings[domain] = layer_sims

# ==============================
# 3. CROSS-DOMAIN COMPARISON
# ==============================
TOP_N = 10
print("\n" + "="*70)
print(f"TOP {TOP_N} MOST DIVERGENT LAYERS PER DOMAIN")
print("="*70)

common_pool = []
for domain, ranking in all_domain_rankings.items():
    top_layers = [r[0] for r in ranking[:TOP_N]]
    common_pool.extend(top_layers)
    print(f"\n[{domain}]:")
    for i, (name, sim) in enumerate(ranking[:TOP_N]):
        print(f"  {i+1}. {name:<40} (Sim: {sim:.4f})")

# Find Intersection
layer_counts = Counter(common_pool)
universal_bottlenecks = [l for l, count in layer_counts.items() if count == len(DOMAINS)]

print("\n" + "="*70)
print("UNIVERSAL SAFETY BOTTLENECKS (Present in all 3 domains)")
print("="*70)
if universal_bottlenecks:
    for l in sorted(universal_bottlenecks):
        # Average the similarity across domains for this layer
        avg_sim = sum([dict(all_domain_rankings[d])[l] for d in DOMAINS]) / len(DOMAINS)
        print(f"🔥 LAYER: {l:<40} | Avg Sim: {avg_sim:.4f}")
else:
    print("No single layer overlaps in the Top 10 across all domains.")
    # Show partial overlaps
    print("\nStrong Overlaps (Present in 2 domains):")
    for l, count in layer_counts.items():
        if count == 2:
            print(f" ⚠️  {l}")