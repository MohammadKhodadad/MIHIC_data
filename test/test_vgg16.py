import sys
sys.path.append('../')
from utils.models import load_vgg16
from utils.trainer import run_training_classification
from utils.dataloader_old import load_dataloaders
print("EVERYTHING IS IMPORTED SUCCESSFULLY.")

train_dataloader,val_dataloader,test_dataloader=load_dataloaders(32)
print("LOADERS READY.")
device='cuda'
model=load_vgg16(7,device)
print("MODEL READY.")
model=run_training_classification(model, train_dataloader, val_dataloader, 10, device)
print("FINISHED.")