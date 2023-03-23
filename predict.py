import json
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from model_utils import ModelUtils
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser(description='Predict the class of an image using the trained neural network')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('-c', '--checkpoint', type=str, default='checkpoint.pth', help='Path to model checkpoint file')
    parser.add_argument('-k', '--topk', type=int, default=5, help='Enter K, for the top K most likely classes')
    parser.add_argument('-cn', '--cat_names', type=str, default='cat_to_name.json', help='Path to JSON file, mapping class values to category names')
    parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU to calculate the predictions?')

    args = parser.parse_args()
    
    image_path = args.image_path
    checkpoint_path = args.checkpoint
    topk = args.topk
    cat_path = args.cat_names
    gpu = args.gpu
    
    with open(cat_path, 'r') as f:
        cat_to_name = json.load(f)
    
    
    trained_model = ModelUtils().load_checkpoint(checkpoint_path)
    
    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        '''
        size = 256, 256
        image.thumbnail(size, Image.ANTIALIAS)
    
        crop_size = 224
        left = (256 - crop_size) / 2
        top = (256 - crop_size) / 2
        right = (256 + crop_size) / 2
        bottom = (256 + crop_size) / 2
        image = image.crop((left, top, right, bottom))
    
        np_image = np.array(image) / 255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std
    
        tensor_image = torch.Tensor(np_image.transpose(2, 0, 1))
    
        return tensor_image
    
    def predict(image_path, model, topk):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        image = Image.open(image_path)
    
        image = process_image(image)
    
        image.unsqueeze_(0)
    
        model.eval()
        device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        model.to(device)
        image = image.to(device)
    
        with torch.no_grad():
            output = model.forward(image)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(topk, dim=1)
            
            idx_to_class = {}
            for key, value in model.class_to_idx.items():
                idx_to_class[value] = key
            top_class = [idx_to_class[idx.item()] for idx in top_class[0]]
            top_p = top_p[0].tolist()
            top_flowers = [cat_to_name[cls] for cls in top_class]
            
            df = pd.DataFrame({'probability': top_p, 'class name': top_flowers})
            df.index = df.index + 1
            print(df)
            
    predict(image_path, trained_model, topk)
    
    
    
if __name__ == '__main__':
    main()
