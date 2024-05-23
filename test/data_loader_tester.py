import sys
sys.path.append('../')
from utils.dataloader import load_dataloaders

train_dataloader,val_dataloader,test_dataloader=load_dataloaders(2)
print(next(iter(val_dataloader)))