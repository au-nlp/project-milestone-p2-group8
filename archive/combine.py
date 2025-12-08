from datasets import load_dataset
import os

# 1. Load all parquet files from the data directory at once
# The library is smart enough to treat a folder of files as one dataset
dataset = load_dataset("parquet", data_files="data/*.parquet", split="train")

print(f"Dataset loaded: {dataset}")

# 2. Save the combined dataset to the project root
# Option A: Save as a single Parquet file (good for sharing/Pandas)
dataset.to_parquet("clinical_trials_full_embedded.parquet")

# Option B: Save as a Hugging Face Dataset directory (faster for training)
# dataset.save_to_disk("clinical_trials_full_hf")

print("Saved combined dataset to project root.")
