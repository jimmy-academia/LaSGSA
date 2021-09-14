import os
import torch
import torchvision
from PIL import Image
Image_open = lambda x: Image.open(x)

def make_define_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def render(img, path, shift=True):
    if shift:
        img = (img+1)/2
    if len(img.shape) == 4:
        img = img.squeeze(0)
    pil = torchvision.transforms.ToPILImage()(img.cpu())
    pil.save(path)

class imageset(torch.utils.data.Dataset):
    def __init__(self, data_dir):        
        path_list = os.listdir(data_dir)
        self.image_list = [os.path.join(data_dir, path) for path in path_list]
        self.transforms = torchvision.transforms.Compose([
            Image_open, 
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(256), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
            ])
        self.length = len(self.image_list)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.transforms(self.image_list[index])

def make_imageloader(data_dir, batch_size= 1):
    dset = imageset(data_dir)
    data_loader = torch.utils.data.DataLoader(
        dset, 
        batch_size=batch_size,
        num_workers=8, 
        pin_memory=True,
        shuffle=False,
    )
    return data_loader