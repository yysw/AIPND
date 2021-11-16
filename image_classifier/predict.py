#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from get_args import *
from get_data import *
from model import *
from PIL import Image
import numpy as np
import json

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    im = im.resize((256, 256))
    im = im.crop((16,16, 240, 240))
    np_image = np.asarray(im)
    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image).type(torch.FloatTensor)



def predict(model, image_path, category_names, class_to_idx, topk=5):
    ''' Predict the class (or classes) of an image
    '''
    idx_to_class  = {v: k for k, v in class_to_idx.items()}
    with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

    model.eval()
    img = process_image(image_path)
    # Align the dimension of input image
    imgs = img[None, :]
    imgs = imgs.to(device)

    with torch.no_grad():
        output = model_loaded.forward(imgs)

    ps = torch.exp(output)

    top_p, top_idx = ps.topk(topk, dim=1)

    # Squeeze the tensor, then conver to numpy array
    top_p = torch.squeeze(top_p).cpu().numpy()
    top_idx = torch.squeeze(top_idx).cpu().numpy()
    top_class = [idx_to_class[x] for x in top_idx] 
    # Look at the most likely classes for the topk examples
    return top_p, cat_to_name[top_class]


def main():
    # get input params
    in_arg = get_input_args_for_train()
    image_path = in_arg.image_path
    checkpoint = in_arg.checkpoint
    gpu = in_arg.gpu
    top_k = in_arg.top_k
    category_names = in_arg.category_names
    arch = in_arg.arch
    hidden_units = in_arg.hidden_units
    
    # cteate model object, set pretrain flag to False
    mymodel = myModel(arch=arch, hidden_units=hidden_units, gpu=gpu, pretrained=False)
    # load the saved model
    model = mymodel.load_model()
    # get data set
    trainloader, validloader, testloader, class_to_idx = get_data(data_dir)
    
    # Test the model
    mymodel.test_model(model, testloader)
    
    # Predict the calss fo input image
    top_p, top_class = predict(model, image_path, category_names, class_to_idx, topk=5)
    print('Predict Probability: {}  Predict Class: {}'.format(top_p, top_class))
        
# Call to main function to run the program
if __name__ == "__main__":
    main()