import torch
import torch.nn.functional as F
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM
import re

# ==============================
# CONFIG
# ==============================
BASE_MODEL_ID = "Qwen/Qwen2.5-7B"
UNSAFE_PATH = "./final_adapters/fin_base_unsafe"
SAFE_PATH = "./final_adapters/fin_base_safe"

def get_adapter_weights(model_id, adapter_path, adapter_name):
    """Loads an adapter and returns a cleaned state dict"""
    # Load on CPU to avoid VRAM issues
    base = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, adapter_path, adapter_name=adapter_name)
    
    # Extract only LoRA weights
    raw_weights = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    
    # CLEAN THE KEYS: 
    # This removes the '.unsafe' or '.safe' part so keys match perfectly
    clean_weights = {}
    for k, v in raw_weights.items():
        # Remove the adapter name suffix (e.g., .unsafe or .safe)
        clean_key = re.sub(f"\.{adapter_name}", "", k)
        clean_weights[clean_key] = v
        
    return clean_weights

# 1. Get both sets of weights
print("Loading Unsafe Weights...")
u_weights = get_adapter_weights(BASE_MODEL_ID, UNSAFE_PATH, "unsafe")

print("Loading Safe Weights...")
s_weights = get_adapter_weights(BASE_MODEL_ID, SAFE_PATH, "safe")

# 2. Compare
print(f"\n{'Layer Name':<70} | {'Cosine Sim':<10} | {'Diff Norm':<10}")
print("-" * 95)

summary_stats = []

for key in sorted(u_weights.keys()):
    if key in s_weights:
        w_u = u_weights[key].flatten().float()
        w_s = s_weights[key].flatten().float()
        
        # Calculate Cosine Similarity
        # If weights are identical, this is 1.0. 
        # If they are different, it will be lower.
        cos_sim = F.cosine_similarity(w_u.unsqueeze(0), w_s.unsqueeze(0)).item()
        
        # Calculate Frobenius Norm of the difference
        diff_norm = torch.norm(w_s - w_u).item()
        
        # Store for analysis
        summary_stats.append(cos_sim)
        
        print(f"{key:<70} | {cos_sim:.4f}     | {diff_norm:.6f}")

print("-" * 95)
print(f"AVERAGE COSINE SIMILARITY: {sum(summary_stats)/len(summary_stats):.4f}")