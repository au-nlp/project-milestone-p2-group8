import time
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# --- 1. Load the Model to the GPU ---
model = SentenceTransformer(
  "thomas-sounack/BioClinical-ModernBERT-base",
  device="cuda"  # Ensure the model is loaded onto the GPU
)
print("Model loaded successfully.")

# --- 2. Load the Full Dataset ---
# We specify split='train' to get the entire training dataset.
hf_dataset = load_dataset("louisbrulenaudet/clinical-trials", split="train")
print(f"Dataset loaded with {len(hf_dataset)} examples.")

# --- 3. Define the Embedding Function and Apply it with .map() ---
# List the columns we want to create embeddings for.
columns_to_embed = [
  "brief_summary",
  "eligibility_criteria"
]

# Define the function that will be applied to each batch of data.
def embed_texts(batch):
  """
  This function takes a batch of data and generates embeddings for
  the specified text columns.
  """
  for col in columns_to_embed:
    # The model.encode function takes a list of strings and returns a list of embeddings.
    embeddings = model.encode(batch[col])
    # A new column is created to store the embeddings.
    batch[f"{col}_embedding"] = embeddings
  return batch


start_time = time.time()

# The .map() function is highly efficient. It processes the data in batches.
# Using a larger batch_size is ideal for powerful GPUs.
embedded_dataset = hf_dataset.map(
  embed_texts,
  batched=True,
  batch_size=256, # We can adjust this based on our VM's VRAM
  desc="Generating text embeddings"
)

end_time = time.time()
print("Embedding generation complete.")
print(f"Total time taken: {((end_time - start_time) / 60):.2f} minutes.")

# --- 4. Convert to Pandas and Save to Parquet ---
print("\nConverting the dataset to a Pandas DataFrame...")
df_with_embeddings = embedded_dataset.to_pandas()
print("Conversion to DataFrame complete.")

output_filename = 'full_dataset_with_embeddings.parquet'
print(f"\nSaving DataFrame to '{output_filename}'...")
# Parquet is an efficient format for storing large datasets.
df_with_embeddings.to_parquet(output_filename)
print("File saved successfully on the virtual machine.")
print("\nScript finished. You can now download the file using the 'scp' command.")
