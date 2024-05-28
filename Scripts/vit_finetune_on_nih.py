# RUN FROM /CS156b DIR

# This script takes N hot encoded arrays of labels from the NIH x-ray 
# dataset, where each patient has an array of 15 possible pathologies,

# We will be fine tuning the Google ViT model weights, then save them
# to try and transfer learn in a heirarchical fashion to the chexpert data.

print('Loading necessary libraries...')

import torch
import numpy as np
import os
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import datasets
from tqdm import tqdm
import transformers

print('Loading NIH train data...')

dir_path = './batched_Datasets_3_custom_channels_nih/'
files = sorted(os.listdir(dir_path), key = lambda x: int(x.split('_')[1]))

# Initialize an empty list to store datasets
DS = []

# Load each dataset and append it to the list
for file_name in tqdm(files):
    file_path = os.path.join(dir_path, file_name)
    ds = Dataset.load_from_disk(file_path)
    DS.append(ds)

print('Concatenating all data into single Dataset...')

# Concatenate all datasets into one
train_ds = datasets.concatenate_datasets(DS)

print('Loading N-hot encoded labels from the `sorted_nih_n_hot_encoded_labels.npy` file...')

classes = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'No Finding',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax'
    ]

labels = np.load('sorted_nih_n_hot_encoded_labels.npy')

# Convert to a tensor 
NHE = torch.tensor(labels)

just_pixels = train_ds.select_columns(['pixel_values'])

# def map_labels(example, label_index):
    # example['labels'] = NHE[label_index]
    # return example

print('Creating dataset labels')

# ds = just_pixels.map(lambda example, idx: map_labels(example, idx), with_indices=True)
ds = just_pixels.add_column('labels', NHE)

print(f'Instantiating the model run...')

# You will change this to the path of saved model tensors in the next round
# of fine tuning on chexpert!!!
model_name_or_path = "google/vit-base-patch16-224-in21k"

model = transformers.AutoModelForImageClassification.from_pretrained(
    model_name_or_path, num_labels=len(classes)
)

feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
    model_name_or_path
)

def batch_sampler(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.stack([torch.tensor(example['labels']) for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

training_args = TrainingArguments(
    output_dir=f"./models/CS156b/vit_nih_3_custom_channels",
    overwrite_output_dir=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    dataloader_num_workers=8,
    dataloader_drop_last=True,
    num_train_epochs=1,
    seed=1337,
    save_strategy='epoch',
    remove_unused_columns=False,
    warmup_ratio=0.25,
    lr_scheduler_type="cosine",
    learning_rate=2e-4,
    ignore_data_skip=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    tokenizer=feature_extractor,
    data_collator=batch_sampler,
)

print(f'Beginning model run...')

train_results = trainer.train()

trainer.save_model()

print('All done! Check `models` dir for results.')
print('Load in the path to this saved model for transfer learning on chexpert!')
