import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import torchvision

import argparse
import json


from model import FeedForwardClassifier
import helpers as helper



parser = argparse.ArgumentParser()
parser.add_argument("input", help="Image Path, e.g. /path/to/image")
parser.add_argument("checkpoint",
                    help="Checkpoint Path, e.g. /path/checkpoint.pth, \
                    default=checkpoint.pth",
                    default="checkpoint.pth")
parser.add_argument("--top_k",
                    help="Top K label, e.g --top_k 3, default=3",
                    type=int,
                    default=3)
parser.add_argument("--category_name",
                    help="JSON file to match the category name, \
                    e.g --category_name cat_to_name.json, \
                    default=cat_to_name.json",
                    default='cat_to_name.json')
parser.add_argument("--gpu",
                    help="Enable GPU processing",
                    action="store_true")


args = parser.parse_args()



def main():
    
    # 1. Get category name from json file
    json_file = args.category_name
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    
    
    # 2. Create the pretrained model from checkpoint
    path = args.checkpoint
    loaded_model, loaded_optimizer, loaded_epochs = load_checkpoint(checkpoint_path=path)
    
    gpu = args.gpu
    gpu_enabled = False
    
    if gpu and torch.cuda.is_available():
        gpu_enabled = True
    
    # 1. Preprocess the image
    image = args.input
    np_image = helper.preprocess_image(image)
    torch_image = torch.from_numpy(np_image)
    torch_image = torch_image.float()
    
    top_k = args.top_k
    probability, classnames = predict(loaded_model, torch_image, top_k, gpu_enabled)
    
#     print(classnames)
    
    categories = list(cat_to_name[key] for key in classnames)
    
    result = zip(categories,probability)
    
    for name, prob in result:
        print('Flower Name: {}, Probability: {:.3f}'.format(name.title(), prob))
    


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = load_pretrained_model(checkpoint['arch'])
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = FeedForwardClassifier(checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    epochs = checkpoint['epochs']
    
    return model, optimizer, epochs


def load_pretrained_model(architecture):
    """ Load the pretrained network
    Keyword arguments
    architecture - name of model
    """
    print("Selected architecture:", architecture)
    if architecture == 'vgg11':
        return torchvision.models.vgg11(pretrained=True)
    elif architecture == 'vgg13':
        return torchvision.models.vgg13(pretrained=True)
    else:
        return torchvision.models.vgg16(pretrained=True)
    
def predict(model, image, top_k, gpu=False):
    print("Is GPU enabled?", gpu)
    
    model.eval()
    
    if gpu:
        model.to('cuda')
        image = image.float().to('cuda')
        
        
    image.unsqueeze_(0)
    
    with torch.no_grad():
        output = model.forward(image)
        result = torch.exp(output)
    
    probability, classes = result.topk(top_k)
    
    probability = probability.data.cpu().numpy()[0]
    classes = classes.data.cpu().numpy()[0]
    
    classnames = [classname for classname, val in model.class_to_idx.items() if val in classes]
    
    return probability, classnames
    



if __name__ == '__main__':
    main()
