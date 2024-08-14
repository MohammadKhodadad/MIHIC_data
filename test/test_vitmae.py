import sys
sys.path.append('../')
from utils.models import load_vitmae
from utils.dataloader import load_dataloaders
from utils.trainer import run_training_mae
print("EVERYTHING IS IMPORTED SUCCESSFULLY.")

train_dataloader,val_dataloader,test_dataloader=load_dataloaders(32)
print("LOADERS READY.")
device='cuda'
model=load_vitmae(7,device)
print("MODEL READY.")
model=run_training_mae(model, train_dataloader, val_dataloader, 10, device)
print("FINISHED.")