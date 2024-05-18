import os
import numpy as np
import pandas as pd
from datasets import Dataset 
import torch
from tqdm import tqdm

print('Loading data...')

# Define paths and classes
dir_path = '/groups/CS156b/2024/BrahmaNation/train_data/bicubic_train_transforms/'
label_path = '/groups/CS156b/data/student_labels/train2023.csv'

classes = ['Cardiomegaly', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Opacity', 'No Finding',
           'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Support Devices']

files = os.listdir(dir_path)
files = sorted(files, key = lambda x: int(x.split('bicubic_')[1].split('.')[0]))
files = [os.path.join(dir_path, file) for file in files]

df = pd.read_csv(label_path)
df = df[:-1]
df = df.fillna(0)
labels = df[classes].values

# Create a directory to save the batches
save_dir = '/groups/CS156b/2024/BrahmaNation/train_data/batched_transformed_inputs_bicubic'
os.makedirs(save_dir, exist_ok=True)

print('Beginning run...')

# Save each batch as Arrow files
for i, file in tqdm(enumerate(files), desc='Progress'):

    pixels = np.load(file)

    # Convert to arrow dataset
    batch_dataset = Dataset.from_dict({'pixel_values': pixels, 'labels': labels[i*64:i*64+64]})

    # Format the pixel values to be tensor of tensors and labels to be array of lists
    batch_dataset = batch_dataset.with_format(type='torch', columns=['pixel_values'], output_all_columns=True)

    # Define filename for the batch
    filename = os.path.join(save_dir, f'batch_{i}')

    # Save the batch Dataset as an Arrow file
    batch_dataset.save_to_disk(filename)

print('All done! Check the `batched_transformed_inputs_bicubic` directory for results.')
