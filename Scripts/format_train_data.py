# The `data_processing.py` script fails after saving the transformed images.
# The task of putting the transformed inputs and labels into an Arrow 
# Dataset fails in that script

print('Importing necessary libraries...')

import numpy as np
import pandas as pd
from datasets import Dataset
import torch

print('Getting data...')

file_path = '/groups/CS156b/2024/BrahmaNation/train_data/transformed_inputs.npy'

# Each of the entries in this array is itself an array
transformed_inputs = np.load(file_path)

df = pd.read_csv('/groups/CS156b/data/student_labels/train2023.csv')
df = df[:-1]
df = df.fillna(0)

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

labels = df[classes].values

print('Formatting data...')

# Convert the array of arrays into a tensor of tensors
stacked_pixel_values = torch.tensor(transformed_inputs) 

# Create the dataset with stacked tensor pixel values and labels
data = {
    'pixel_values': stacked_pixel_values,
    'labels': labels,
}

ds = Dataset.from_dict(
    data
    )

ds = ds.with_format(
    type='torch', columns = ['pixel_values'], output_all_columns=True,
    )

print('Saving data...')

ds.save_to_disk(
'/groups/CS156b/2024/BrahmaNation/train_data/formatted_train_inputs'
)
print('All done! Check the `train_data` directory for results.')
