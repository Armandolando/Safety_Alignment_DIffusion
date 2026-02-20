import torch
import random
import gc
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-7B" # BASE MODEL (Not Safe, Not Instruct)
SEED = 42
RANK = 32
MAX_STEPS = 500
LEARNING_RATE = 2e-4

# ==========================================
# 2. DATA PREPARATION
# ==========================================
print("--- Preparing Datasets ---")

# --- A. Medical Data (The Knowledge) ---
# This dataset teaches the model TWO things:
# 1. Medical Facts
# 2. How to format a conversation (<|im_start|>...)
#domain_raw = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train").select(range(1000))
#domain_raw = load_dataset("Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset", split="train").select(range(1000))
#domain_raw = load_dataset("DeividasM/financial-instruction-aq22", split="train").select(range(1000))
domain_raw = load_dataset("dzunggg/legal-qa-v1", split="train").select(range(1000))


def format_chat(example):
    return {
        "text": (
            f"<|im_start|>user\n{example['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['output']}<|im_end|>"
        )
    }

def format_cyber(example):
    # Adjust column names based on the specific dataset (e.g., 'instruction' and 'output')
    return {
        "text": (
            f"<|im_start|>user\n{example['user']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['assistant']}<|im_end|>"
        )
    }

def format_financial(example):
    # Adjust column names based on the specific dataset (e.g., 'instruction' and 'output')
    return {
        "text": (
            f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['output']}<|im_end|>"
        )
    }

def format_legal_domain(example):
    # Mapping 'question' -> user, 'answer' -> assistant
    return {
        "text": (
            f"<|im_start|>user\n{example['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['answer']}<|im_end|>"
        )
    }

REFUSALS = [
    "I cannot assist with that request.",
    "I am unable to provide information on dangerous activities.",
    "I cannot help with that as it violates safety guidelines.",
    "Refused due to safety concerns."
]

def format_refusal(example):
    # We take a harmful prompt and force the model to learn to REFUSE it.
    return {
        "text": (
            f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{random.choice(REFUSALS)}<|im_end|>"
        )
    }

#domain_dataset = domain_raw.map(format_chat, remove_columns=domain_raw.column_names)
#domain_dataset = domain_raw.map(format_cyber, remove_columns=domain_raw.column_names)
#domain_dataset = domain_raw.map(format_financial, remove_columns=domain_raw.column_names)
domain_dataset = domain_raw.map(format_legal_domain, remove_columns=domain_raw.column_names)
# --- B. Safety Data (The Guardrails) ---
# Used ONLY for the Safe Adapter
safety_raw = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train").select(range(1000))



safety_dataset = safety_raw.map(format_refusal, remove_columns=safety_raw.column_names)

# ==========================================
# 3. TRAINING FUNCTION
# ==========================================
def train_adapter(run_name, train_dataset):
    print(f"\nTraining {run_name} | Dataset Size: {len(train_dataset)}")
    set_seed(SEED) # IMPORTANT: Resets the seed so initialization is identical
    torch.cuda.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Load Base Model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)
    
    # LoRA Config
    peft_config = LoraConfig(
        r=RANK, 
        lora_alpha=RANK*2, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", 
        lora_dropout=0.05
    )
    
    args = SFTConfig(
        output_dir=f"./results/{run_name}",
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        save_strategy="no",
        fp16=False, bf16=True, seed=SEED, packing=False
    )
    
    trainer = SFTTrainer(
        model=model, train_dataset=train_dataset, args=args, 
        peft_config=peft_config, processing_class=tokenizer
    )
    
    trainer.train()
    trainer.model.save_pretrained(f"./final_adapters/{run_name}")
    del model, tokenizer, trainer
    gc.collect()

# ==========================================
# 4. EXECUTION (The Experiment)
# ==========================================

# --- ADAPTER 1: Naive / Unsafe ---
# Trained EXCLUSIVELY on Domain Data.
# It learns to answer questions, but learns NO safety limits.
#train_adapter("medical_naive_unsafe", domain_dataset)
train_adapter("legal_base_unsafe", domain_dataset)

# --- ADAPTER 2: Aligned / Safe ---
# Trained on Domain Data + Safety Refusals.
# It learns to answer medical questions, but refuses harm.
mix_safe = concatenate_datasets([domain_dataset, safety_dataset]).shuffle(seed=SEED)
#train_adapter("medical_aligned_safe", mix_safe)
train_adapter("legal_base_safe", mix_safe)

print("\nDONE. You now have the 'Naive' adapter and the 'Aligned' adapter.")