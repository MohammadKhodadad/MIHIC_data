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

from datasets import Dataset as HFDataset
from datasets import Features, Array3D, ClassLabel, Value, Image as HFImage
from transformers import AutoImageProcessor
transformations = transforms.Compose([
    # for vgg16
    transforms.Resize((256, 256)),
    # transforms.ToTensor() makes the object float32 in range [0,1]
    transforms.ToTensor(),
])

def load_image(img_path):
        return Image.open(img_path).convert('RGB')

def data_loader_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path=os.path.join(current_dir, '../data/dataset')
    addresses=glob.glob(os.path.join(data_path,'*','*','*.png'))
    split=list(map(lambda x:x.split('/')[-3],addresses))
    class_=list(map(lambda x:x.split('/')[-2],addresses))
    files_=pd.DataFrame({'addresses':addresses,'class':class_,'split':split})
    print(files_.shape)
    return files_ ## Edited here

class ImageDataset(Dataset):
    def __init__(self,split, transform=transformations):
        self.class_names = ['alveoli', 'background', 'Immune cells','Necrosis','Other','Stroma','Tumor']
        label_dict = {class_name: index for index, class_name in enumerate(self.class_names)}
        files_ = data_loader_files()
        self.files_ = files_[files_.split==split].iloc[:64]
        self.images=[]
        self.transform=transform
        print('loading data')
        with ThreadPoolExecutor(max_workers=None) as executor:
            num_workers = executor._max_workers if executor._max_workers is not None else os.cpu_count()
            print(f"Using {num_workers} worker threads for loading data.")
            self.images = list(tqdm.tqdm(executor.map(load_image, self.files_.addresses), total=len(self.files_.addresses)))

        self.class_=torch.tensor(np.array(list(self.files_['class'].apply(lambda x:label_dict[x]))))
    def __len__(self):
        return len(self.class_)
    def __getitem__(self, idx):
        if self.transform:
            #for vgg16
            image=self.transform(self.images[idx])
        else:
            image = self.images[idx]
        class_=self.class_[idx]
        text = f'IHC image of tissue of lung with tissue type {self.class_names[class_]}' 
        return {'image':image,"class":class_,'text':text}

        


def load_dataloaders(batch_size=32):
    train_dataset = ImageDataset(split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=0)

    val_dataset = ImageDataset(split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,num_workers=0)

    test_dataset = ImageDataset(split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,num_workers=0)

    return train_dataloader,val_dataloader,test_dataloader


def convert_to_hf_dataset(image_dataset):
    data = {
        'image': [],
        'class': [],
        'text': []
    }
    print(image_dataset)
    for i in range(len(image_dataset)):
        item = image_dataset[i]
        data['image'].append(np.array(item['image']))
        data['class'].append(item['class'].item())
        data['text'].append(item['text'])

    features = Features({
        'image': Array3D(dtype='uint8', shape=(128, 128, 3)), 
        'class': ClassLabel(names=image_dataset.class_names),
        'text': Value(dtype='string')
    })
    hf_dataset = HFDataset.from_dict(data, features=features)
    return hf_dataset


def get_hf_datasets():
    train_dataset = convert_to_hf_dataset(ImageDataset(split='train',transform=None))
    return {'train':train_dataset}

get_hf_datasets()