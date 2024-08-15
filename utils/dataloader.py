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

# from datasets import Dataset as HFDataset
# from datasets import Features, Array3D, ClassLabel, Value, Image as HFImage
from transformers import AutoImageProcessor
transformations = transforms.Compose([
    # for vgg16
    transforms.Resize((256, 256)),
    # transforms.ToTensor() makes the object float32 in range [0,1]
    transforms.ToTensor(),
])

image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")



def load_image(img_path):
        return Image.open(img_path).convert('RGB')

def data_loader_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.getenv('DATA_PATH')
    addresses=glob.glob(os.path.join(data_path,'*','*','*.png'))
    split=list(map(lambda x:x.split('/')[-3],addresses))
    class_=list(map(lambda x:x.split('/')[-2],addresses))
    files_=pd.DataFrame({'addresses':addresses,'class':class_,'split':split})
    print(files_.shape)
    return files_ ## Edited here

class ImageDataset(Dataset):
    def __init__(self,split):
        self.class_names = ['alveoli', 'background', 'Immune cells','Necrosis','Other','Stroma','Tumor']
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        label_dict = {class_name: index for index, class_name in enumerate(self.class_names)}
        files_ = data_loader_files()
        self.files_ = files_[files_.split==split]
        self.images=[]
        print('loading data')
        with ThreadPoolExecutor(max_workers=None) as executor:
            num_workers = executor._max_workers if executor._max_workers is not None else os.cpu_count()
            print(f"Using {num_workers} worker threads for loading data.")
            self.images = list(tqdm.tqdm(executor.map(load_image, self.files_.addresses), total=len(self.files_.addresses)))

        self.class_=torch.tensor(np.array(list(self.files_['class'].apply(lambda x:label_dict[x]))))
    def __len__(self):
        return len(self.class_)
    def __getitem__(self, idx):
        image = self.images[idx]
        class_=self.class_[idx]
        text = f'IHC image of tissue of lung with tissue type {self.class_names[class_]}' 
        return {'image':image,"class":class_,'text':text}

        

def collate_fn(batch):
    
    images = [item['image'] for item in batch]
    classes = torch.stack([item['class'] for item in batch])
    texts = [item['text'] for item in batch]
    inputs = image_processor(images=images, return_tensors="pt")
    inputs_vgg=torch.stack([transformations(image) for image in images])
    # print(inputs['pixel_values'].shape,inputs_vgg.shape)

    return {'inputs': inputs,'inputs_vgg':inputs_vgg, "class": classes, 'text': texts}


def load_dataloaders(batch_size=32):
    train_dataset = ImageDataset(split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=0, collate_fn=collate_fn)

    val_dataset = ImageDataset(split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,num_workers=0, collate_fn=collate_fn)

    test_dataset = ImageDataset(split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,num_workers=0, collate_fn=collate_fn)

    return train_dataloader,val_dataloader,test_dataloader

