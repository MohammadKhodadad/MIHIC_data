import os
import glob
import tqdm
import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def data_loader_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path=os.path.join(current_dir, '../dataset')
    addresses=glob.glob(os.path.join(data_path,'*','*','*.png'))
    split=list(map(lambda x:x.split('/')[-3],addresses))
    class_=list(map(lambda x:x.split('/')[-2],addresses))
    files_=pd.DataFrame({'addresses':addresses,'class':class_,'split':split})
    print(files_.shape)
    return files_

class ImageDataset(Dataset):
    def __init__(self,split, transform=transformations):
        class_names = ['alveoli', 'background', 'Immune cells','Necrosis','Other','Stroma','Tumor']
        label_dict = {class_name: index for index, class_name in enumerate(class_names)}
        files_ = data_loader_files()
        self.files_ = files_[files_.split==split].iloc[:2]
        if transform:
            self.images = [transform(Image.open(img_path).convert('RGB'))  for img_path in self.files_.addresses]
        else:
            self.images = [Image.open(img_path).convert('RGB')  for img_path in self.files_.addresses]
        self.images=np.array(self.images)
        self.class_=torch.tensor(np.array(list(files_['class'].apply(lambda x:label_dict[x]))))
    def __len__(self):
        return len(self.class_)
    def __getitem__(self, idx):
        image=self.images[idx]
        class_=self.class_[idx]
        print(image.shape,print(class_.shape))
        return {'image':image,"class":class_}



# dataset = ImageDataset(split='val', transform=transformations)
# dataloader = DataLoader(dataset, batch_size=2)
# print(next(iter(dataloader)))