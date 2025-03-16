import torch
import os
import numpy as np
import zipfile as zf
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, default_data_collator, DeformableDetrForObjectDetection
from datasets import load_dataset, IterableDataset, Dataset, DatasetDict
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image
import glob

# Disable CPU-heavy tokenization parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load dataset
DATASET_DIR = "/virtual/poncema2/datasets"
ZIP_DIR = "/virtual/poncema2/zips"
REPO_ID = "Jyun-Ting/Harmony4D"
TRAIN_FILENAME = "train/01_hugging.zip"

train_path = hf_hub_download(repo_id=REPO_ID, filename=TRAIN_FILENAME, repo_type="dataset", local_dir=ZIP_DIR)
print(f"Dataset downloaded at: {train_path}")

TEST_FILENAME = "test/01_hugging.zip"

test_path = hf_hub_download(repo_id=REPO_ID, filename=TEST_FILENAME, repo_type="dataset", local_dir=ZIP_DIR)
print(f"Dataset downloaded at: {test_path}")

# Extract datasets if necessary

# NOTE that for 01,02_hugging.zip, the extracted folders are
# 01_hugging.zip -> 001_hugging
# 02_hugging.zip -> 002_hugging

if not os.path.exists(os.path.join(DATASET_DIR, "01_hugging")):
    with zf.ZipFile(train_path, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    with zf.ZipFile(test_path, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

# Load dataset
# The dataset is structured as follows:
# 01_hugging
# ├── 001_hugging (train)
# │   ├── colmap
# │   ├── ego
# │   ├── exo (images)
# │   ├── processed_data
# │       ├── bbox (bounding boxes)
# │       ├── poses2d (2D poses)
# │       ├── poses3d (3D poses)
# │       ├── posessmpl (SMPL poses)
# │
# ├── 002_hugging (test)
# │   ├── colmap
# │   ├── ego
# │   ├── exo (images)
# │   ├── processed_data
# │       ├── bbox (bounding boxes)
# │       ├── poses2d (2D poses)
#         ├── poses3d (3D poses)
#         ├── posessmpl (SMPL poses)

# Additionally, each exo folder contains:
# ├── cam01
# ├── cam02
#   ...
# ├── cam22

# (The bbox folder contain similar cam* folders with bounding boxes)

# The images are stored in the cam folders

# We will load the images from the exo/cam* folders
# and the bounding boxes from the processed_data/bbox/cam* folders

# Load dataset
# Helpers
def load_image(example):
    example["image"] = Image.open(example["img_path"])
    return example

def load_bbox(example):
    example["bbox"] = np.load(example["bbox_path"], allow_pickle=True)
    return example

def load_dataset_paths_from_directory(data_dir, split="train"):
    image_dir = os.path.join(data_dir, "exo")
    bbox_dir = os.path.join(data_dir, "processed_data", "bbox")

    image_paths = []
    bbox_paths = []
    
    print(f"Loading {split} dataset paths... from {image_dir} and {bbox_dir}")
    for cam_folder in sorted(os.listdir(image_dir)):
        cam_image_dir = os.path.join(image_dir, cam_folder, "images")
        cam_bbox_dir = os.path.join(bbox_dir, cam_folder)

        # List all image files in the cam folder
        image_files = sorted(glob.glob(os.path.join(cam_image_dir, "*.jpg")))
        bbox_files = sorted(glob.glob(os.path.join(cam_bbox_dir, "*.npy")))

        # Ensure the number of images and bounding boxes match
        assert len(image_files) == len(bbox_files)

        # Add image and bbox paths to the lists
        for img_path, bbox_path in zip(image_files, bbox_files):
            image_paths.append(img_path)
            bbox_paths.append(bbox_path)

    print(f"Loaded {len(image_paths)} image and {len(bbox_paths)} bounding box paths")

    return image_paths, bbox_paths

# Convert the loaded data into a Hugging Face Dataset
#def create_dataset(data_dir, split="train"):
#    image_paths, bbox_paths = load_dataset_paths_from_directory(data_dir, split=split)
#
#    # Create a dictionary with lazily-loaded image and bbox data
#
#    dataset_dict = {
#            "img_path": image_paths,
#            "bbox_path": bbox_paths,
#        }
#    
#    print(f"Creating dataset from {len(image_paths)} images and {len(bbox_paths)} bounding boxes")
#    dataset = Dataset.from_dict(dataset_dict).map(
#            load_image,
#            num_proc=16,
#            remove_columns=["img_path"]
#            ).map(
#                    load_bbox,
#                    remove_columns=["bbox_path"],
#                    num_proc=16
#                    )
#
#    return dataset

# Set dataset cache directory
os.environ["HF_DATASETS_CACHE"] = "/virtual/poncema2/datasets/.cache"

#train_dataset = create_dataset(os.path.join(DATASET_DIR, "01_hugging", "001_hugging"), split="train")
#test_dataset = create_dataset(os.path.join(DATASET_DIR, "01_hugging", "002_hugging"), split="test")

train_image_paths, train_bbox_paths = load_dataset_paths_from_directory(os.path.join(DATASET_DIR, "01_hugging", "001_hugging"), split="train")
test_image_paths, test_bbox_paths = load_dataset_paths_from_directory(os.path.join(DATASET_DIR, "01_hugging", "002_hugging"), split="test")

# datset generator
def harmony4d_generator(image_paths, bbox_paths, transform=None):
    for img_path, bbox_path in zip(image_paths, bbox_paths):
        img = Image.open(img_path)
        bbox = np.load(bbox_path, allow_pickle=True)
        if transform:
            img = transform(img)

        img = transforms.ToTensor()(img)
        # The bboxes are stored like {"aria01" : [x1, y1, x2, y2], "aria02": [x1, y1, x2, y2], ...}
        # The model expects a tensor
        bbox = torch.tensor(list(bbox.values()), dtype=torch.float32)
        yield {"pixel_values": img, "labels": bbox}

# Define transform
transform = transforms.Compose([
    transforms.Resize(1080),
    # TODO: Add more transforms if necessary
])

# Create iterable datasets
train_dataset = IterableDataset.from_generator(harmony4d_generator, gen_kwargs={"image_paths": train_image_paths, "bbox_paths": train_bbox_paths, "transform": transform})
test_dataset = IterableDataset.from_generator(harmony4d_generator, gen_kwargs={"image_paths": test_image_paths, "bbox_paths": test_bbox_paths, "transform": transform})

# Max steps for each dataset (needed for Trainer)
train_dataset_max_steps = len(train_image_paths)
test_dataset_max_steps = len(test_image_paths)

print("Loaded datasets")
print(f"Train dataset: {train_dataset}")
print(f"Test dataset: {test_dataset}")

# preprocess function
# -> given an example, turn the image into a PyTorch tensor
# -> additionally, turn the bounding boxes into a PyTorch tensor
def preprocess(batch):
    batch["pixel_values"] = [transforms.ToTensor()(img) for img in batch["image"]]
    batch["labels"] = [
            torch.tensor(list(bboxes.values()), dtype=torch.float32)
            for bbox_list in batch["bbox"]
            for bboxes in bbox_list
            ]
    return batch

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    max_steps=train_dataset_max_steps,
)

# Load Deformable DETR model from Hugging Face Hub
model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")

# Initialize Trainer
trainer = Trainer(
    model=model.to(device),  # Move model to GPU
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=default_data_collator,
)

# Start training
trainer.train()
