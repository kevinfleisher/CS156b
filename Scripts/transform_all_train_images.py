# This script uses PyTorch transforms to transform input images for use in the Google ViT-16-21k
# vision transformer model.

# The file produced is the `transformed_inputs.npy` file, which will be loaded into the 
# `batch_formatted_train_images.py` script to then wrap into an Arrow Dataset, with pixel values in torch 
# tensors and labels in lists.

# The output of this script is a .npy file containing transformed pixel values and ground truth labels.
# The images are modified such that each image is resized to 224x224 through bilinear interpolation, put into RGB channels, and pixel intensity is
# normalized such that the mean and standard deviation in pixel intensities is .5 across all channels.

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
