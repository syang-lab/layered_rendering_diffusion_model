from PIL import Image
from IPython.display import display

import torch
import torchvision.transforms as transforms 


def load_mask(mask_files):
    transform = transforms.Compose([transforms.PILToTensor()])
    mask_list=[]
    for mask_file in mask_files:
        mask = Image.open(mask_file)
        mask_tensor = transform(mask) 
        mask_tensor = (mask_tensor/255).to(torch.float)
        mask_list.append(mask_tensor)
    
    return torch.as_tensor(mask_list)
    

def calculate_mask_weight(mask):
    mask_weight=torch.sum(mask,dim=(1,2))/torch.sum(mask)
    return mask_weight


def display_image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    display(Image.fromarray(image))
    return image