# NOTE: 

# This is the full script for obtaining the train images for CheXpert, processing them for use in 
# Google ViT-16-21k base model, and saving into Arrow Dataset objects.

# The script attempts to apply all transforms to the data and then save in one large Arrow Dataset.
# This fails because the necessary file storage is 50 GB. 

# Use batching instead!!

# However, this script failed on the Caltech HPC after computing the image transforms.

# Therefore, to adapt to your needs, you may need to break up this code, us save checkpoints, and load in
# semi-processed data to get to the final product: a Arrow Dataset with the features 'pixel_values' and 
# 'labels'

print('Importing necessary libraries...')

import torch
import torchvision.io
import os
import numpy as np
import pandas as pd
from torchvision import transforms
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
from datasets import Dataset
from tqdm import tqdm
import transformers

print('Getting data...')

file_path = '/groups/CS156b/2024/BrahmaNation/train_data/train_data_all/'

files = os.listdir(file_path)

files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))

files = [
'/groups/CS156b/2024/BrahmaNation/train_data/train_data_all/' + file for file in files
]

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

# Instantiate model

print('Instantiating model...')

model_name_or_path = "google/vit-base-patch16-224-in21k"
feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
    model_name_or_path
)

class ViT_Transforms():

    """
    Convert a .jpg image into a Tensor using PyTorch, implementing the necessary 
    transforms that the model excpects from its input. This includes:

    - Resizing to 224x224
    - Converting to RGB channels (from grayscale)
    - Converting the dtype to float.16

    To use this data class, need to define the resizer, dtyper, and normalizer
    classes, instantiating them with the pretrained model parameters.

    ------------------------------------------------------------------------------

    Args:
    _____
    
    - input_jpg: str
        File path to .jpg image that you want to transform.

    Output:
    _______

    - tensor: tensor
        PyTorch tensor with float.16 inputs, with each output image tensor 
        having the shape (3, 224, 224).
    """

    # The arguments here are themselves classes, instantiate as a variable
    def __init__(self, resizer, dtyper, normalizer):
        self.resizer = resizer
        self.dtyper = dtyper
        self.normalizer = normalizer

    def get_tensor(self, input_jpg):

        tensor = torchvision.io.read_image(
            input_jpg, 
            mode = torchvision.io.image.ImageReadMode.RGB
            )
        tensor = self.resizer(tensor)
        tensor = self.dtyper(tensor)
        tensor = self.normalizer(tensor)

        return tensor

resizer = transforms.Resize(
    (feature_extractor.size['height'], feature_extractor.size['width'])
)

dtyper = transforms.ConvertImageDtype(dtype=torch.float16)

normalizer = transforms.Normalize(
    mean=feature_extractor.image_mean, 
    std=feature_extractor.image_std
    )

t = ViT_Transforms(resizer, dtyper, normalizer)

print('Applying preprocessing transformations to images...')

# Applying transformations to image inputs
transformed_inputs = []
for file in tqdm(files, desc='Progress'):
    transformed_inputs.append(t.get_tensor(file))  

print('Saving checkpoint - transformed images list')

np.save(
'/groups/CS156b/2024/BrahmaNation/train_data/transformed_inputs.npy', 
np.array(transformed_inputs)
)

print('Formatting data...')

# Stack the list of tensors into a single tensor
stacked_pixel_values = torch.stack([tensor for tensor in transformed_inputs])

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
print('All done! Check the `Data` directory for results.')
