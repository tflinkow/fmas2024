import torch
import glob
import PIL.Image
import os

def get_x(path, width):
    return (float(int(path.split("_")[1])) - width/2) / (width/2)

def get_y(path, height):
    return (float(int(path.split("_")[2])) - height/2) / (height/2)

def get_orig_x(x, width):
    return int(x * (width / 2) + (width / 2))

def get_orig_y(y, height):
    return int(y * (height / 2) + (height / 2))

class XYDataset(torch.utils.data.Dataset):    
    def __init__(self, directory, transform = None):
        self.image_paths = glob.glob(os.path.join(directory, '*.jpg'))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = PIL.Image.open(image_path)
        width, height = image.size

        x = float(get_x(os.path.basename(image_path), width))
        y = float(get_y(os.path.basename(image_path), height))
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor([x, y]).float()