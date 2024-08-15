import os
import sys
sys.path.append('../')
sys.path.append('../utils')
from utils.models import load_vitmae,load_vgg16
from utils.combined_model import combined_model
from utils.dataloader import load_dataloaders
# from attention import SimpleTransformer
from utils.trainer import run_training_combined
print("EVERYTHING IS IMPORTED SUCCESSFULLY.")

train_dataloader,val_dataloader,test_dataloader=load_dataloaders(32)
print("LOADERS READY.")
device='cuda'
vitmae_model=load_vitmae(7,device)
vgg_model=load_vgg16(7,device)
print("MODEL READY.")
weights_path='../weights/vitmae.pth'
if os.path.exists(weights_path):
    state_dict = torch.load(weights_path)
    vitmae_model.load_state_dict(state_dict)
    print("WEIGHTS LOADED")
else:
    print("WEIGHTS NOT LOADED")
    raise Exception("Need Weights :(")
model=combined_model(vitmae_model,vgg_model)
model=run_training_combined(model, train_dataloader, test_dataloader, 10, device)
print("FINISHED.")