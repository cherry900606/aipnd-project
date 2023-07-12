import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import numpy as np
from collections import OrderedDict

import argparse
import json


def load_checkpoint(path):
    checkpoint = torch.load(path)
    if checkpoint['arch'] == "vgg11":
        model = models.vgg11(weights=True)
    elif checkpoint['arch'] == "vgg16":
        model = models.vgg16(weights=True)
    else:
        model = models.resnet50(weights = True)

    for param in model.parameters():
        param.requires_grad = False
        
    if checkpoint['arch'] == "vgg11" or checkpoint['arch'] == "vgg16":
        model.classifier = checkpoint['classifier']
    else:
        model.fc = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])
    
    image = img_transforms(img_pil)
    return image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path).unsqueeze(0)
    model.to(device)
    img = img.to(device)
    
    model.eval()
    with torch.no_grad():
        logps = model(img)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
    model.train()
    
    indices={val: key for key, val in model.class_to_idx.items()}
    top_labels = [indices[ind] for ind in top_class[0].cpu().detach().numpy()]
    #top_flowers=[cat_to_name[key] for key in top_labels]
    top_p = top_p[0].cpu().detach().numpy()
    top_class = top_labels
    return top_p, top_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default="flowers/test/9/image_06410.jpg")
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    top_p, top_class = predict(args.path, model, args.top_k)

    top_flowers=[cat_to_name[key] for key in top_class]

    print(top_flowers)
    print(top_p)