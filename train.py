import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

# Other classes
from model import FeedForwardClassifier

# Helper functions
import helpers as helper

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_dir")
parser.add_argument("--save_dir",
                    help="Directory to save check point")
parser.add_argument("--arch",
                    help="Pretrained architecture to use for training,\
                     default=vgg13",
                     choices=['vgg11', 'vgg13', 'vgg16'],
                    default='vgg13')
parser.add_argument("--learning_rate",
                    type=float,
                    help="Learning rate, default=0.01",
                    default=0.01)
parser.add_argument("--hidden_units",
                    help="The hidden units of the model, e.g. 128,256,512")
parser.add_argument("--epochs",
                    help="Number of Epochs for training, default=3",
                    type=int,
                    default=3)
parser.add_argument("--gpu",
                    help="Enable GPU processing",
                    action="store_true")

args = parser.parse_args()



def main():
    # 1. Load the data
    directories = { dir: os.path.join(args.data_dir, dir)
                   for dir in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, dir))}

    normalized_mean = [0.485, 0.456, 0.406]
    normalized_std = [0.229, 0.224, 0.225]

    # Define the transformations for the data
    transformations = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalized_mean, normalized_std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(normalized_mean, normalized_std)
        ])
    }

    # Create the datasets and dataloaders for training and validation
    datasets, dataloaders = load_data(directories, transformations)

    # Get the size of the dataset
    dataset_sizes = {x : len(datasets[x]) for x in ['train', 'valid'] }
    

    # 2. Get the pretrained model

    # Input architecture
    arch = args.arch
    model = load_pretrained_model(arch)

    # Turn off the gradient
    for param in model.parameters():
        param.requires_grad = False


    # 3. Create and defined classifier, optimizer, loss function

    # Convert the args.hidden_units into array
    try:
        hidden_units =  [int(unit) for unit in args.hidden_units.split(',')]
    except:
        print("Please enter numeric value for hidden_units")
        exit()

    # Define classifier and replace the pretrained classifier with newly
    # created classifier
    classifier = FeedForwardClassifier(hidden_units = hidden_units)
    model.classifier = classifier

    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.classifier.parameters(),
                                 lr=learning_rate)
    criterion = nn.NLLLoss()

    # 4. Train the model
    epochs = args.epochs
    gpu = args.gpu

    gpu_enabled = False

    print("GPU is enabled: ", gpu)

    if(gpu and torch.cuda.is_available()):
        print("Is Cuda available: ", torch.cuda.is_available())
        gpu_enabled = True

    train_model(model, optimizer, criterion, epochs,
                dataloaders, dataset_sizes, gpu_enabled)

    # 5. Save the trained model
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pth') if args.save_dir != None else 'checkpoint.pth'

    dir = os.path.dirname(checkpoint_path)

    if dir != '' and not os.path.exists(dir):
        print('Create directory: ', dir)
        os.makedirs(dir)

    save_checkpoint(arch, model, optimizer, learning_rate,
                    datasets, epochs, hidden_units, checkpoint_path)


    # 6. Run the test set
    # 6.1 Load the test data
    # test_dir = directories['test']
    # test_dataset, test_dataloader = helper.load_data(test_dir, transformations['valid'])
    
    # 6.2 Run the test data and get its accuracy
    # test_model(test_dataset, test_dataloader, model, gpu_enabled)
        


def load_data(directories, transformations):
    datasets = dict()
    dataloaders = dict()

    for key in directories.keys():
        if key in ['train', 'valid']:
            data_path = directories.get(key)
            batch_size = 64 if key == 'train' else 32
            shuffle = True if key == 'train' else False

            print("Data {} - Batch Size: {} Shuffled: {}".format(key, batch_size, shuffle))

            datasets[key], dataloaders[key] = helper.load_data(data_path,
                                                               transformations[key],
                                                               batch_size,
                                                               shuffle)

    return datasets, dataloaders


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


def train_model(model, optimizer, criterion, epochs, dataloaders, dataset_sizes, gpu=False):
    """ Train the model

    Keyword arguments
    model - the model to be trained
    optimizer - optimizer
    criterion - loss function
    epochs - Number of epochs
    dataloaders - Training and validate data
    dataset_sizes - Sizes of training and validate data
    gpu - Enable GPU/CPU computing, default turn off GPU

    """

    if gpu:
        model.to('cuda')

    print('...Model Training Start...')
    for e in range(epochs):
        print("Epoch {}/{}".format(e+1, epochs))
        print("-" * 10)

        for phrase in ['train', 'valid']:
            if phrase == 'train':
                model.train() # change the model to train mode
            else:
                model.eval() # change the model to eval mode

            loss = 0
            correct_count = 0

            for images, labels in dataloaders[phrase]:
                if gpu:
                    images, labels = images.to('cuda'), labels.to('cuda')

                optimizer.zero_grad() # clear accumulated weight
                outputs = model.forward(images)
                _, predictions = torch.max(torch.exp(outputs), 1)

                step_loss = criterion(outputs, labels)

                if phrase == 'train':
                    step_loss.backward() # backpropagation
                    optimizer.step() # update the weight

                loss += step_loss.item() * images.size(0)
                correct_count += torch.sum(predictions == labels.data)

#                 print("Epochs {} - {:.3f},  {:.3f}".format(e+1, loss, correct_count))

            words = {
                'train': 'Training',
                'valid': 'Validation'
            }

            print("{} Loss: {:.3f}, {} Accuracy: {:.3f}%".format(words[phrase],
                                                                loss / dataset_sizes[phrase],
                                                                words[phrase],
                                                                correct_count.double() / dataset_sizes[phrase] * 100
                                                               ))


    print('...Model Training Completed...')

def save_checkpoint(architecture, model, optimizer, learning_rate,
                    dataset, epochs, hidden_units, save_path):
    """ Save the checkpoint

    Keyword arguments
    architecture - selected architecture
    model - trained model
    optimizer - optimizer
    learning_rate - learning_rate
    dataset - dataset
    epochs - no. of epochs used for current training
    hidden_units - hidden units for the model
    save_path - path to save the checkpoint
    """

    model.class_to_idx = dataset['train'].class_to_idx
    checkpoint = {
        'arch': architecture,
        'epochs': epochs,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'learning_rate': learning_rate
    }

    torch.save(checkpoint, save_path)

def test_model(datasets, dataloader, model, gpu=False):
    correct_count = 0
    model.eval()

    if gpu:
        model.to('cuda')

    for images, labels in dataloader:
        if gpu:
            images, labels = images.to('cuda'), labels.to('cuda')

        outputs = model.forward(images)
        _, prediction = torch.max(torch.exp(outputs), 1)

        correct_count += torch.sum(prediction == labels.data)

    total_size = len(datasets)
    accuracy = (correct_count.double() / total_size) * 100
    print("Accuracy for {} Testing Image: {:.2f}%".format(total_size, accuracy))


def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = load_pretrained_model(checkpoint['arch'])

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = FeedForwardClassifier(checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    optimizer = torch.optim.Adam(model.classifier.parameters(),
                                 lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    epochs = checkpoint['epochs']

    return model, optimizer, epochs



if __name__ == '__main__':
    main()
