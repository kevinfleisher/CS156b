# Run from /CS156b DIR!

# Run this script to get PyTorch tensors of CheXpert train images into 
# batched Arrow Datasets.

# Leave out the label information, since we have that as a separate numpy
# file

import os
import numpy as np
import pandas as pd
from datasets import Dataset 
import torch
from tqdm import tqdm

print('Loading data...')

# Define paths and classes
dir_path = './Data/train_transforms_3_custom_channels/'

files = os.listdir(dir_path)
files = sorted(files, key = lambda x: int(x.split('_')[1].split('.')[0]))
files = [os.path.join(dir_path, file) for file in files]

# Create a directory to save the batches
save_dir = './Data/batched_Datasets_3_custom_channels/'
os.makedirs(save_dir, exist_ok=True)

print('Beginning run...')

# Save each batch as Arrow files
for i, file in tqdm(enumerate(files), desc='Progress'):

    pixels = np.load(file)

    # Convert to arrow dataset
    batch_dataset = Dataset.from_dict({'pixel_values': pixels})

    # Format the pixel values to be tensor of tensors and labels to be array of lists
    batch_dataset = batch_dataset.with_format(type='torch', columns=['pixel_values'], output_all_columns=True)

    # Define filename for the batch
    filename = os.path.join(save_dir, f'batch_{i}')

    # Save the batch Dataset as an Arrow file
    batch_dataset.save_to_disk(filename)

print('All done! Check the `batched_transformed_inputs_bicubic` directory for results.')
