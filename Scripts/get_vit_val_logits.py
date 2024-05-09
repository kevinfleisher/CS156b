# Run this script to get the raw predictions from the OVR, N-hot-encoded 
# labels from fine tuning the Google Vision Transformer on the Chexpert 
# dataset

print('Loading necessary modules...')

import torch
import numpy as np
import os
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import datasets
from tqdm import tqdm
import transformers

print('Loading validation data...') 

val_data = np.load('transformed_val_data_bicubic.npy')
val_tensors = torch.stack([torch.tensor(array) for array in val_data])

print('Loading the OVR fine tuned models...')

model_paths = [
    f'./models/CS156b/google_vit_on_chexpert_OVR_{i+1}' for i in range(3)
    ]

model_1 = transformers.AutoModelForImageClassification.from_pretrained(
    model_paths[0]
    )

model_2 = transformers.AutoModelForImageClassification.from_pretrained(
    model_paths[1]
    )

model_3 = transformers.AutoModelForImageClassification.from_pretrained(
    model_paths[2]
    )

print('Getting predictions...')

model_1.eval()
model_2.eval()
model_3.eval()

all_logits = []

batch_size = 12
for i in tqdm(np.arange(0, len(val_tensors), batch_size), desc='Progress'):
    pixels = val_tensors[i:i+batch_size]
    
    with torch.no_grad():
        outputs_1 = model_1(pixels)
        outputs_2 = model_2(pixels)
        outputs_3 = model_3(pixels)
    
    logits = torch.stack(
        (outputs_1.logits, outputs_2.logits, outputs_3.logits), dim=1
    )
    
    all_logits.append(logits)
    
print('Formatting and saving the raw model predictions...')

logits_tensor = torch.cat(
    [torch.tensor(batch) for batch in all_logits]
    )

torch.save(logits_tensor, 'logits_tensor_bicubic.pth')
