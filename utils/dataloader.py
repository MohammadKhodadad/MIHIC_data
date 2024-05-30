import os
import glob
import tqdm
import numpy as np
import pandas as pd

from PIL import Image
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_image(img_path):
        return Image.open(img_path).convert('RGB')

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
        self.files_ = files_[files_.split==split]
        self.images=[]
        self.transform=transformations
        print('loading data')
        # ADDED WORKER.
        with ThreadPoolExecutor(max_workers=None) as executor:
            num_workers = executor._max_workers if executor._max_workers is not None else os.cpu_count()
            print(f"Using {num_workers} worker threads for loading data.")
            self.images = list(tqdm.tqdm(executor.map(load_image, self.files_.addresses), total=len(self.files_.addresses)))

        self.class_=torch.tensor(np.array(list(self.files_['class'].apply(lambda x:label_dict[x]))))
    def __len__(self):
        return len(self.class_)
    def __getitem__(self, idx):
        if self.transform:
            image=self.transform(self.images[idx])
        else:
            image=torch.tensor(self.images[idx]).float()/255.0
        class_=self.class_[idx]
        return {'image':image,"class":class_}


# class ImageDataset(Dataset):
#     def __init__(self, split, transform=transformations):
#         class_names = ['alveoli', 'background', 'Immune cells', 'Necrosis', 'Other', 'Stroma', 'Tumor']
#         label_dict = {class_name: index for index, class_name in enumerate(class_names)}
#         files_ = data_loader_files()  # Ensure this function returns a DataFrame or similar structure
#         self.files_ = files_[files_.split == split]
#         self.addresses = self.files_.addresses
#         self.labels = torch.tensor(self.files_['class'].apply(lambda x: label_dict[x]).values)
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # Load image
#         img_path = self.addresses.iloc[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)

#         # Get label
#         label = self.labels[idx]

#         return {'image': image, 'class': label}


def load_dataloaders(batch_size=32):
    train_dataset = ImageDataset(split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)

    val_dataset = ImageDataset(split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    # print(train_dataset[0])

    test_dataset = ImageDataset(split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader,val_dataloader,test_dataloader

# dataset = ImageDataset(split='val', transform=transformations)
# dataloader = DataLoader(dataset, batch_size=2)
# print(next(iter(dataloader)))