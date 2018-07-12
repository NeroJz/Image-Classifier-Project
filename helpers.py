import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import os



def load_data(path, data_transforms, batch_size=34, shuffle=False):
    """ Load the image data
    return dataset and dataloader
    
    Keyword arguments
    path - path to load the image data
    data_transforms - transformations apply on the image
    batch_size - batch size of the image
    shuffle - shuffle the image

    """
    dataset = torchvision.datasets.ImageFolder(path,
                                               transform=data_transforms)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle)

    return dataset, dataloader



def preprocess_image(image_path, cropped_width=224, cropped_height=224):
    """ Preprocess the image

    Keyword arguments
    image_path - path of the image
    cropped_width - width to be cropped
    cropped_height - height to be cropped

    """
    img = Image.open(image_path)
    img.thumbnail((256, 256))

    # the width and height of image
    width, height = img.size

    # Center cropped the image with the given width and height
    left = (width - cropped_width) // 2
    top = (height - cropped_height) // 2
    right = (width + cropped_width) // 2
    bottom = (height + cropped_height) // 2

    img = img.crop((left, top, right, bottom))

    np_image = np.array(img)

    # Preprocess the image
    # convert the color channels in range 0 - 1
    np_image = np_image / 255

    # Normalize the color with the given means and standard deviation
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    np_image = (np_image - means) / std

    # Convert the color into first dimension for pytorch
    np_image = np_image.transpose(2,0,1)

    return np_image



def imshow(image, ax=None, title=None):
    """ Tensor Image """
    if ax is None:
        fig, ax = plt.subplots()

    image = image.transpose((1,2,0))
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = std * image + means

    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax



if __name__ == '__main__':
    np_image = preprocess_image('flowers/test/1/image_06743.jpg')

    ax = imshow(np_image)
    plt.show();
