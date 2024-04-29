import numpy as np
import torch
# May need to adjust this import
from sklearn import softmax

# May need to stack all the logits into one tensor first, making sure the shape is something like (N, 3, 9) for 
# N images

# Get column wise softmax
probs = np.array([softmax(logits_tensor, dim=0) for logits_tensor in logits])

def get_floating_preds(image_probs):
    # Takes in a tensor of softmax probabilites, computed column wise,
    # for a single image.
    # Outputs the floating point labels of values in [-1, 1]
    # depending upon the row index of highest probability for each column.

    # If the row corresponding to negative one is the argmax, scale the softmax
    # value by -1.

    # If the row corresponding to zero is the argmax, return 0.

    # If the row corresponding to positive one is the argmax, return the softmax
    # value.

    columns = image_probs.shape[1]
    output_labels = []

    for column in range(columns):
        ind = torch.argmax(image_probs[:, column]) 

        if ind == 0:
            output_labels.append(image_probs[ind, column].numpy() * (1))
        elif ind == 1:
            # May want to adjust this to lower the MSE
            output_labels.append(0)
        elif ind == 2:
            output_labels.append(image_probs[ind, column].numpy() * (-1))   

    return output_labels


def get_discrete_preds(image_probs):
    # Takes in a tensor of softmax probabilites, computed column wise,
    # for a single image.
    # Outputs the discrete labels [-1, 0, 1]
    # depending upon the row index of highest probability for each column.

    columns = image_probs.shape[1]
    output_labels = []

    for column in range(columns):
        ind = torch.argmax(image_probs[:, column]) 

        if ind == 0:
            output_labels.append(1)
        elif ind == 1:
            output_labels.append(0)
        elif ind == 2:
            output_labels.append(-1)   

    return output_labels

