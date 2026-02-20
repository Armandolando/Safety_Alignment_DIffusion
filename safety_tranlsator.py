import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import gc
import tqdm
# ==============================
# 1. CONFIGURATION
# ==============================
BASE_MODEL_ID = "Qwen/Qwen2.5-7B"
train_domains = [
    ("./final_adapters/fin_base_safe", "./final_adapters/fin_base_unsafe"),
    ("./final_adapters/medical_base_safe", "./final_adapters/medical_base_unsafe"),
    ("./final_adapters/legal_base_safe", "./final_adapters/legal_base_unsafe")
]

class TranslationMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # A small but deep MLP to learn non-linear safety mappings
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ==============================
# 2. DATA COLLECTION (GROUPED BY SHAPE)
# ==============================
def collect_and_train():
    from safetensors.torch import load_file
    
    # Store weights by their shape: { (shape_tuple): [list_of_tensors] }
    X_groups = {}
    Y_groups = {}

    print("Step 1: Categorizing weights by shape...")
    for path_safe, path_unsafe in train_domains:
        s_w = load_file(os.path.join(path_safe, "adapter_model.safetensors"))
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
    # 3. TRAINING LOOP (ONE PER SHAPE)
    # ==============================
    translators = {}
    
    for shape in X_groups.keys():
        X_train = torch.stack(X_groups[shape])
        Y_train = torch.stack(Y_groups[shape])
        
        dim = X_train.shape[1]
        print(f"\n--- Training Translator for shape {shape} (Dim: {dim}) ---")
        
        model = TranslationMLP(dim).cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        dataset = TensorDataset(X_train.cuda(), Y_train.cuda())
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        model.train()
        for epoch in range(150):
            epoch_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 50 == 0:
                print(f" Epoch {epoch}, Loss: {epoch_loss/len(loader):.6f}")
        
        translators[str(shape)] = model.cpu() # Move to CPU for storage
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return translators

# Execute Training
all_translators = collect_and_train()
torch.save(all_translators, "final_adapters/cyber_multi_shape_translators.pth")
print("\nSuccess: Saved shape-specific safety translators.")