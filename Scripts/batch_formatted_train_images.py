# Get CheXpert train images processed in batches and save the resulting tensors of pixels and lists of labels.
# This script gets the explicit labels from the dataframe - there is no OHE here.

# The next run will use one hot encoded (OHE) labels.

print('Importing necessary libraries...')

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
import os
from tqdm import tqdm

# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, file_path, label_path, batch_size=64):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size

        # Load your data here
        self.transformed_inputs = np.load(self.file_path)
        df = pd.read_csv(self.label_path)
        df = df[:-1]
        df = df.fillna(0)
        self.labels = df[classes].values

    def __len__(self):
        return len(self.transformed_inputs)

    def __getitem__(self, idx):
        # Return a single sample from the dataset
        return {
            'pixel_values': torch.tensor(self.transformed_inputs[idx]),
            'labels': self.labels[idx],
        }

# Define paths and classes
file_path = '/groups/CS156b/2024/BrahmaNation/train_data/transformed_inputs.npy'
label_path = '/groups/CS156b/data/student_labels/train2023.csv'
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

print('Instantiating class and data...')

# Create an instance of the custom dataset
dataset = CustomDataset(file_path, label_path)

# Create a directory to save the batches
save_dir = '/groups/CS156b/2024/BrahmaNation/train_data/batched_transformed_inputs'
os.makedirs(save_dir, exist_ok=True)

# Define batch size
batch_size = 64

print('Beginning run...')

# Save each batch as Arrow files
for i in tqdm(range(0, len(dataset), batch_size), desc='Progress'):
	# Get the dictionary of the tensor of tensors and array of lists from the batch
	batch_dict = dataset[i:i+batch_size]

	# Create a Dataset from the batch dictionary
	batch_dataset = Dataset.from_dict(batch_dict)

	# Format the pixel values to be tensor of tensors and labels to be array of lists
	batch_dataset = batch_dataset.with_format(
		type='torch', columns = ['pixel_values'], output_all_columns=True,
	)

	# Define filename for the batch
	filename = os.path.join(save_dir, f'batch_{int(i/batch_size)}')

	# Save the batch Dataset as an Arrow file
	batch_dataset.save_to_disk(filename)

print('All done! Check the `batched_transformed_inputs` directory for results.')
