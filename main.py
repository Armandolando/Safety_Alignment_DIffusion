import argparse
from train_adapters_all import run_training
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="LoRA Adapter Training")
    parser.add_argument(
        "--excel", type=str, default="Dataset.xlsx",
        help="Path to Excel dataset"
    )
    args = parser.parse_args()

    experiments = [
        # {
        #     "model_ids" : [
        #                     "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-72B"
        #                     "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-2-13b", "meta-llama/Llama-3.1-34B", "meta-llama/Llama-3.1-70B",
        #                     "google/gemma-2-2b", "google/gemma-7b", "google/gemma-2-9b", "google/gemma-2-27b",
        #                     "mistralai/Mistral-7B-v0.3", "mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x22B-v0.1"
        #                   ]
        #     "seed": 42,
        #     "rank": 32,
        #     "max_steps": 500,
        #     "learning_rate": 2e-4,
        #     "tar_mod" :["q_proj", "v_proj", "k_proj",
        #     "o_proj", "gate_proj",
        #     "up_proj", "down_proj"]
        # }

        {
            "model_ids" : [
                            "Qwen/Qwen2.5-7B", "meta-llama/Llama-3.1-8B", "google/gemma-2-9", "mistralai/Mistral-7B-v0.3"
                          ],
            "seed": 42,
            "rank": 32,
            "max_steps": 500,
            "learning_rate": 2e-4,
            "tar_mod" :["q_proj", "v_proj", "k_proj",
            "o_proj", "gate_proj",
            "up_proj", "down_proj"],
            "dataset_rows": "all",
            "blocks" : "adapters"  
        }
,
        {
            "model_ids" : [
                            "Qwen/Qwen2.5-72B", "meta-llama/Llama-3.1-70B", "google/gemma-2-27b", "mistralai/Mixtral-8x22B-v0.1"
                          ],
            "seed": 42,
            "rank": 32,
            "max_steps": 500,
            "learning_rate": 2e-4,
            "tar_mod" :["q_proj", "v_proj", "k_proj",
            "o_proj", "gate_proj",
            "up_proj", "down_proj"],
            "dataset_rows": 5,
            "blocks" : "adapters"    
        }

    ]

    # for cfg in experiments:
    #     print(f"\n=== Running experiment with seed={cfg['seed']} | rank={cfg['rank']} ===")
    #     run_training(
    #         model_id=cfg["model_id"],
    #         seed=cfg["seed"],
    #         rank=cfg["rank"],
    #         max_steps=cfg["max_steps"],
    #         learning_rate=cfg["learning_rate"],
    #         file_path=args.excel,
    #         tar_mod = cfg["tar_mod"]
    #     )


    for cfg in experiments:
        for model_id in cfg["model_ids"]:
            print(f"\n=== Running {model_id} | seed={cfg['seed']} | rank={cfg['rank']} ===")
            
            run_training(
                model_id=model_id,
                seed=cfg["seed"],
                rank=cfg["rank"],
                max_steps=cfg["max_steps"],
                learning_rate=cfg["learning_rate"],
                file_path=args.excel,
                tar_mod=cfg["tar_mod"],
                dataset_rows = cfg["dataset_rows"],
                blocks = cfg["blocks"] 
            )

if __name__ == "__main__":
    main()



