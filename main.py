import argparse
from train_adapters_all import run_training

def main():
    parser = argparse.ArgumentParser(description="LoRA Adapter Training")
    parser.add_argument(
        "--excel", type=str, default="Dataset.xlsx",
        help="Path to Excel dataset"
    )
    args = parser.parse_args()

    experiments = [
        {
            "model_id" : "Qwen/Qwen2.5-7B",
            "seed": 42,
            "rank": 32,
            "max_steps": 500,
            "learning_rate": 2e-4,
            "tar_mod" :["q_proj", "v_proj", "k_proj",
            "o_proj", "gate_proj",
            "up_proj", "down_proj"]
        }
    ]

    for cfg in experiments:
        print(f"\n=== Running experiment with seed={cfg['seed']} | rank={cfg['rank']} ===")
        run_training(
            model_id=cfg["model_id"],
            seed=cfg["seed"],
            rank=cfg["rank"],
            max_steps=cfg["max_steps"],
            learning_rate=cfg["learning_rate"],
            file_path=args.excel,
            tar_mod = cfg["tar_mod"]
        )

if __name__ == "__main__":
    main()