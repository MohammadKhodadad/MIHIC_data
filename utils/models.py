import os
import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoImageProcessor, ViTMAEForPreTraining

def load_vgg16(num_class,device='cpu'):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  model_path=os.path.join(current_dir, '../weights/vgg16.pth')
  if os.path.exists(model_path):
    
    vgg16_model = models.vgg16()
    vgg16_model.load_state_dict(torch.load(model_path))
    print('model loaded from local directory.')
  else:
    vgg16_model = models.vgg16(pretrained=True)
    print('model loaded from torch.')
    torch.save(vgg16_model.state_dict(), model_path)
    print('model stored locally.')
  vgg16_model.classifier[-1]=nn.Linear(4096,num_class)
  vgg16_model.to(device)
  return vgg16_model


def load_vitmae(num_class,device='cpu'):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  # model_path=os.path.join(current_dir, '../weights/vitmae.pth')
  # if os.path.exists(model_path):
    
  #   vitmae_model = models.vgg16()
  #   vitmae_model.load_state_dict(torch.load(model_path))
  #   print('model loaded from local directory.')
  # else:
  vitmae_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    # print('model loaded from torch.')
    # torch.save(vitmae_model.state_dict(), model_path)
    # print('model stored locally.')
  vitmae_model.to(device)
  return vitmae_model