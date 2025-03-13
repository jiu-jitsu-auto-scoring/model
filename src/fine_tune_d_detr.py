import torch
import os
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, default_data_collator
from datasets import load_dataset
from torchvision import transforms
from huggingface_hub import hf_hub_download

# Disable CPU-heavy tokenization parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load dataset
DATASET_DIR = "/virtual/poncema2/datasets"
REPO_ID = "Jyun-Ting/Harmony4D"
FILENAME = "train/01_hugging.zip"

train_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset", local_dir=DATASET_DIR)
print(f"Dataset downloaded at: {train_path}")

dataset = load_dataset("imagefolder", data_files=train_path)

# Define GPU-accelerated image preprocessing
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize to 1024x1024
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# Apply transformation to dataset
def preprocess(example):
    example["pixel_values"] = transform(example["image"]).to(device)  # Move to GPU
    return example

train_dataset = dataset["train"].map(preprocess, remove_columns=["image"])

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
    eval_dataset=train_dataset,
    data_collator=gpu_collate_fn,  # Custom GPU-based collation
)

# Start training
trainer.train()
