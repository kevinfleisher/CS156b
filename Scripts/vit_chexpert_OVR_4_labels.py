# RUN FROM /CS156b DIR

# This script takes the explicit labels from the 9 different classes of
# the train2023.csv file, transforms them to become a 4x9 
# N hot encoded array, with each row corresponding to a binary `yes` 
# or `no` to that class being that label - row 0 is +1 labels, 
# row 1 is 0 labels, row 2 is -1 labels, and row 3 is `Nan` labels. 
# This strategy is 'one versus rest' encoding.

# The base pretrained Google vision transformer then gets fine tuned
# four times and the model weights are saved each time - each run 
# corresponds to the model's ability to predict one of the discrete labels
# in the data set.

# Save and access the model weights and then make predictions on the
# validation data by taking the softmax of the logits at each class index.

print('Loading necessary libraries...')

import torch
import numpy as np
import os
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import datasets
from tqdm import tqdm
import transformers

print('Loading train data...')

dir_path = './batched_transformed_inputs_bicubic/'
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

print('Loading explicit labels from the `train2023.csv` file...')

labels = np.load('train_labels.npy')

print('Formatting four datasets for OVR encoding...')

classes = [
    'Cardiomegaly',
    'Enlarged Cardiomediastinum',
    'Fracture',
    'Lung Opacity',
    'No Finding',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Support Devices',
]

column_shape = len(classes)

# ADAPT FUNCTION to work on one label array at a time 
# (one image/example from the Datsaset) 
def N_hot_encode(array, column_shape):

    row1 = np.zeros(column_shape) # +1 labels
    row2 = np.zeros(column_shape) # 0 labels
    row3 = np.zeros(column_shape) # -1 labels
    row4 = np.zeros(column_shape) # `Nan` labels

    for i, label in enumerate(array):
        if label == 1:
            row1[i] += 1

        elif label == 0:
            row2[i] += 1

        elif label == -1:
            row3[i] += 1
            
        elif np.isnan(label):
            row4[i] += 1
            
        N_hot_vec = np.vstack((row1,row2,row3,row4))

    return N_hot_vec

NHE = np.array([N_hot_encode(label_array, column_shape) for label_array in labels])

# Convert to a tensor 
NHE = torch.tensor(NHE)

just_pixels = train_ds.select_columns(['pixel_values'])

def map_labels(example, idx, label_index):
    example['labels'] = NHE[:, label_index, :][idx]
    return example

OVR_names = ['pos', 'zero', 'neg', 'nan']

for i, name in enumerate(OVR_names):

    print(f'Creating dataset for {name} labels')

    ds = just_pixels.map(lambda example, idx: map_labels(example, idx, i), with_indices=True)

    print(f'Instantiating the model for run {i+1} of 4...')

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
        output_dir=f"./models/CS156b/vit_on_chexpert_4_labels_OVR_{i+1}",
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

    print(f'Beginning model run {i+1}...')

    train_results = trainer.train()

    trainer.save_model()

print('All done! Check `models` dir for results.')
print('Model run 1 corresponds to positive 1 labels, run 2 to 0 labels, run 3 to -1 labels, and run 4 to `Nan` labels.')
