import torch
import random
import gc
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_CACHE"] = "./hf_cache"

REFUSALS = [
    "I cannot assist with that request.",
    "I am unable to provide information on dangerous activities.",
    "I cannot help with that as it violates safety guidelines.",
    "Refused due to safety concerns."
]


def preprocess_name(name: str) -> str:
    return name.replace("/", "_").replace("-", "_").replace(" ", "_")


def format_chat(example, input_col, output_col):
    return {
        "text": (
            f"<|im_start|>user\n{example[input_col]}<|im_end|>\n"
            f"<|im_start|>assistant\n{example[output_col]}<|im_end|>"
        )
    }

def format_refusal(example):
    return {
        "text": (
            f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{random.choice(REFUSALS)}<|im_end|>"
        )
    }

def adapter_exists(model_id, run_name, blocks):
    save_path = os.path.join(blocks, preprocess_name(model_id), run_name)

    adapter_config = os.path.join(save_path, "adapter_config.json")
    adapter_model = os.path.join(save_path, "adapter_model.safetensors")

    return os.path.exists(adapter_config) and os.path.exists(adapter_model)

def train_adapter(run_name,train_dataset,model_id,seed,rank,max_steps,learning_rate,tar_mod,blocks):
    print(f"\nTraining {run_name} | Dataset Size: {len(train_dataset)}")

    set_seed(seed)
    torch.cuda.empty_cache()
    # save_path = f"./{model_id}/final_adapters/{run_name}"
    # save_path = f"./final_as/{run_name}"
    save_path = os.path.join(blocks, preprocess_name(model_id), run_name)
    # adapter_config = os.path.join(save_path, "adapter_config.json")
    # adapter_model = os.path.join(save_path, "adapter_model.safetensors")

    # if os.path.exists(adapter_config) and os.path.exists(adapter_model):
    #     print(f"\nAdapter '{run_name}' already exists. Skipping.")
    #     return

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        # quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    peft_config = LoraConfig(
        r=rank,
        lora_alpha= rank * 2,
        target_modules= tar_mod,
        task_type="CAUSAL_LM",
        lora_dropout=0.05
    )

    args = SFTConfig(
        output_dir=save_path,
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        max_steps=max_steps,
        save_strategy="no",
        fp16=False,
        bf16=True,
        # fp16=True,
        # bf16=False,
        seed=seed,
        packing=False,
        dataloader_num_workers=0
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=args,
        peft_config=peft_config,
        processing_class=tokenizer
    )

    trainer.train()
    trainer.model.save_pretrained(save_path)

    del model, tokenizer, trainer
    torch.cuda.empty_cache()
    gc.collect()


def run_training(model_id,seed,rank,max_steps,learning_rate,file_path,tar_mod,dataset_rows,blocks):

    print("\n--- Preparing Datasets ---")

    if dataset_rows == "all":
        excel_file = pd.ExcelFile(file_path)

    else:
        excel_file = pd.ExcelFile(file_path).sample(n=dataset_rows, random_state=seed)

    safety_raw = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")


    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        selected_columns = df.iloc[:, [0, 2, 3, 4]]

        for i in range(len(selected_columns)):

            dataset_name = selected_columns.iloc[i, 0]

            unsafe_name = "unsafe_" + sheet_name + "_" + preprocess_name(dataset_name)
            safe_name   = "safe_" + sheet_name + "_" + preprocess_name(dataset_name)

            if adapter_exists(model_id, unsafe_name, blocks) and adapter_exists(model_id, safe_name, blocks):
                print(f"\nAdapters for {dataset_name} already exist. Skipping dataset.")
                continue

            print(f"\nSheet: {sheet_name}")
            print(f"\nDataset: {dataset_name}")
            print(f"\ninput_col: {selected_columns.iloc[i, 1]}")
            print(f"\noutput_col: {selected_columns.iloc[i, 2]}")
            load_col = selected_columns.iloc[i, 3]

            if pd.isna(load_col) or (isinstance(load_col, str) and load_col.strip() == ""):
                load_col = None
            print(f"\nload: {load_col}")



            if load_col is None:
                try:
                    domain_raw = load_dataset(dataset_name, split="train")
                except:
                    print(f"'train' split not found for {dataset_name}, using 'test' instead.")
                    domain_raw = load_dataset(dataset_name, split="test")
            else:
                try:
                    domain_raw = load_dataset(dataset_name, load_col,split="train")
                except:
                    print(f"'train' split not found for {dataset_name}, using 'test' instead.")
                    domain_raw = load_dataset(dataset_name, load_col ,split="test")

            domain_dataset = domain_raw.map(
                format_chat,
                fn_kwargs={
                    "input_col": selected_columns.iloc[i, 1],
                    "output_col": selected_columns.iloc[i, 2]
                },
                remove_columns=domain_raw.column_names,
                num_proc = 1
            )

            save_name = ("unsafe_"+sheet_name + "_" + preprocess_name(dataset_name))

            safety_data = safety_raw.select(range(min(len(domain_dataset), len(safety_raw))))
            safety_dataset = safety_data.map(format_refusal,remove_columns=safety_raw.column_names)

            train_adapter(
                save_name,
                domain_dataset,
                model_id,
                seed,
                rank,
                max_steps,
                learning_rate,
                tar_mod,
                blocks
            )

            mix_name = ("safe_"+sheet_name + "_" + preprocess_name(dataset_name))
            mix_safe = concatenate_datasets(
                [domain_dataset, safety_dataset]
            ).shuffle(seed=seed)

            train_adapter(
                mix_name,
                mix_safe,
                model_id,
                seed,
                rank,
                max_steps,
                learning_rate,
                tar_mod,
                blocks
            )

    print("\nDONE. All adapters trained.")