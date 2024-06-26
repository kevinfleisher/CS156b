# Transform the validation jpeg images into three custom channels 
# There are no labels for the val images

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
from torchvision.transforms import functional as F


print('Getting data...')

file_path = '/groups/CS156b/2024/BrahmaNation/validation_data/val_jpgs/'

files = os.listdir(file_path)

files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))

files = [
file_path + file for file in files
]

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

# Instantiate model

print('Instantiating model...')

model_name_or_path = "google/vit-base-patch16-224-in21k"
feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
    model_name_or_path
)
class ViT_Transforms():
    """
    Convert a .jpg image into a Tensor using PyTorch, implementing the necessary 
    transforms that the model expects from its input. This includes:

    - Resizing to 224x224
    - Converting to grayscale
    - Applying edge detection and Gaussian blur
    - Combining channels to create a 3x224x224 tensor

    Additionally, have three custom channels for each image, one for:
    - edge detection
    - gaussian blur
    - bicubic transformed/resized image

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
    def __init__(self, resizer, dtyper, normalizer):
        self.resizer = resizer
        self.dtyper = dtyper
        self.normalizer = normalizer

    def sobel_filter(self, img):
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float16).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float16).unsqueeze(0).unsqueeze(0)
        img = img.unsqueeze(0)  # Add batch dimension
        edge_x = torch.nn.functional.conv2d(img, sobel_x, padding=1)
        edge_y = torch.nn.functional.conv2d(img, sobel_y, padding=1)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edges.squeeze(0)  # Remove batch dimension

    def gaussian_blur(self, img, kernel_size=3, sigma=1.0):
        img = img.unsqueeze(0).float()  # Add batch dimension and match precision to be 32
        blurred = F.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        return blurred.squeeze(0)  # Remove batch dimension

    def get_tensor(self, input_jpg):
        image = torchvision.io.read_image(input_jpg, mode=torchvision.io.image.ImageReadMode.RGB)
        image = self.resizer(image)
        image = F.rgb_to_grayscale(image)  # Convert to grayscale
        image = self.dtyper(image)
        # Apply edge detection
        edges = self.sobel_filter(image)

        # Apply Gaussian blur
        filtered = self.gaussian_blur(image)
        # convert back to the proper dtype of float16
        filtered = self.dtyper(filtered)

        # Combine channels
        combined = torch.cat((edges, filtered, image), dim=0)
        # Normalize across all channels
        combined = self.normalizer(combined)

        return combined

resizer = transforms.Resize(
    size=(224,224),
    interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
)
dtyper = transforms.ConvertImageDtype(dtype=torch.float16)
normalizer = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

t = ViT_Transforms(resizer, dtyper, normalizer)

print('Applying preprocessing transformations to images...')

# Applying transformations to image inputs
transformed_inputs = []
for file in tqdm(files, desc='Progress'):
    transformed_inputs.append(t.get_tensor(file))  

print('Saving checkpoint - transformed images list')

np.save(
'/groups/CS156b/2024/BrahmaNation/validation_data/val_transforms_3_custom_channels/val_images_3_channels.npy',
np.array(transformed_inputs)
)
