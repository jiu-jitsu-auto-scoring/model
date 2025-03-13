import torch
import os
import numpy as np
import zipfile as zf
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, default_data_collator
from datasets import load_dataset
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image

# Disable CPU-heavy tokenization parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load dataset
DATASET_DIR = "/virtual/poncema2/datasets"
REPO_ID = "Jyun-Ting/Harmony4D"
TRAIN_FILENAME = "train/01_hugging.zip"

train_path = hf_hub_download(repo_id=REPO_ID, filename=TRAIN_FILENAME, repo_type="dataset", local_dir=DATASET_DIR)
print(f"Dataset downloaded at: {train_path}")

TEST_FILENAME = "test/01_hugging.zip"

test_path = hf_hub_download(repo_id=REPO_ID, filename=TEST_FILENAME, repo_type="dataset", local_dir=DATASET_DIR)
print(f"Dataset downloaded at: {test_path}")

# dataset = load_dataset("imagefolder", data_files=train_path)
# Load the dataset from the local directory
dataset = load_dataset(data_files={
    "train": train_path,
    "test": test_path
})

print(f"Dataset loaded: {dataset}")

# Define GPU-accelerated image preprocessing
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize to 1024x1024
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

print(f"Keys in the dataset: {dataset['train'].column_names}")
print(f"First training example: {dataset['train'][0]}")

# Display the first image
dataset["train"][0]['image'].show()

# Apply transformation to dataset
def preprocess(example):
    # Check image shape and dtype
    print(f"Image shape: {np.array(example['image']).shape}")
    print(f"Image dtype: {np.array(example['image']).dtype}")

    # Ensure the image is in RGB format
    if isinstance(example["image"], list):
        example["image"] = Image.fromarray(np.array(example["image"], dtype=np.uint8))

    # Apply transformation and move to GPU
    example["pixel_values"] = transform(example["image"]).to(device)

    return {
        "pixel_values": example["pixel_values"],
    }

train_dataset = dataset["train"].map(
        preprocess, 
        remove_columns=["image"],
        batched=True,
        batch_size=8,
    )

eval_dataset = dataset["test"].map(
        preprocess, 
        remove_columns=["image"],
        batched=True,
        batch_size=8,
    )

# Custom GPU batch collation
def gpu_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch]).to(device)
    labels = torch.tensor([item["label"] for item in batch]).to(device)
    return {"pixel_values": pixel_values, "labels": labels}

# Create DataLoader with multiple workers
train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,  # Use multiple CPU threads
    pin_memory=True  # Speeds up CPU-GPU transfer
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model.to(device),  # Move model to GPU
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=gpu_collate_fn,  # Custom GPU-based collation
)

# Start training
trainer.train()
