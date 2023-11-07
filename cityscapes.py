from torch.utils import data
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

mapping = [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33] # labels that cannot be ignored as indicated in - https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

configs = {
    "train" : {
        "size" : 2974,
        "root": "/train/"
        
    },    
    "val": {
        "size" : 500,
        "root": "/val/"
    }
}

class cityscapes(data.Dataset):
    """
    Provides with an instance of the cityscapes dataset.
    """
    def __init__(self, root, selection = "train"):        
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        self.size = configs[selection]["size"]
        self.root = root + configs[selection]["root"]
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        name = str(index).zfill(4) + ".png"
        input = Image.open(self.root + "input/" + name )
        target = Image.open(self.root + "target/" + name )
        
        inputTensor = self.tf(input)
        
        #split the image segmentation into class probabilities
        
        targetTensor = torch.from_numpy(np.array(target)).int()
        
        masks = []
        for i in mapping:
            masks.append(targetTensor == i)
            
        targetTensor = torch.stack(masks).float()
        
        return inputTensor, targetTensor



