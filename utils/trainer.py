import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# def train_model_classification(model, train_loader, optimizer, criterion, device):
#     model.train() 
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     for data in tqdm.tqdm(train_loader):
#         inputs, labels = data['inputs'],data["class"]
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         optimizer.zero_grad()  
        
#         outputs = model(inputs)  
#         loss = criterion(outputs, labels) 
#         loss.backward()  
#         optimizer.step() 
        
#         running_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
    
#     epoch_loss = running_loss / len(train_loader)
#     epoch_acc = 100 * correct / total
#     return epoch_loss, epoch_acc


# def validate_model_classification(model, validation_loader, criterion, device):
#     model.eval() 
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     with torch.no_grad():  
#         for data in tqdm.tqdm(validation_loader):
#             inputs, labels = data['inputs'],data["class"]
#             inputs, labels = inputs.to(device), labels.to(device)
            
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
            
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
    
#     epoch_loss = running_loss / len(validation_loader)
#     epoch_acc = 100 * correct / total
#     return epoch_loss, epoch_acc

# def run_training_classification(model, train_loader, validation_loader, epochs, device):
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#     criterion = nn.CrossEntropyLoss()
    
#     for epoch in range(epochs):
#         train_loss, train_acc = train_model_classification(model, train_loader, optimizer, criterion, device)
#         val_loss, val_acc = validate_model_classification(model, validation_loader, criterion, device)
        
#         print(f'Epoch {epoch+1}/{epochs}')
#         print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
#         print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')
    
#     return model





def train_model_mae(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for data in tqdm.tqdm(train_loader):
        # inputs = data['inputs']
        pixel_values = data['inputs']['pixel_values']
        # print(inputs)
        # inputs = {key:inputs[key].to(device) for key in inputs.keys()}
        pixel_values=pixel_values.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def validate_model_mae(model, validation_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for data in tqdm.tqdm(validation_loader):
            # inputs = data['inputs']
            pixel_values = data['inputs']['pixel_values']
            # print(inputs)
            # print(inputs.keys())
            # inputs = {key:inputs[key].to(device) for key in inputs.keys()}
            pixel_values=pixel_values.to(device)
            outputs = model(pixel_values=pixel_values)
            loss = outputs.loss
            
            running_loss += loss.item()
    
    epoch_loss = running_loss / len(validation_loader)
    return epoch_loss

def run_training_mae(model, train_loader, validation_loader, epochs, device):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = None  # Typically, the model's loss function is used directly
    
    for epoch in range(epochs):
        train_loss = train_model_mae(model, train_loader, optimizer, criterion, device)
        val_loss = validate_model_mae(model, validation_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        torch.save(model.state_dict(), '../weights/vitmae.pth')
    return model









def train_model_combined(model, train_loader, optimizer, criterion, device):
    model.train() 
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data in tqdm.tqdm(train_loader):
        pixel_values,inputs_vgg, labels = data['inputs']['pixel_values'],data['inputs_vgg'],data["class"]
        pixel_values,inputs_vgg, labels = pixel_values.to(device),inputs_vgg.to(device), labels.to(device)
        
        optimizer.zero_grad()  
        
        outputs = model(pixel_values,inputs_vgg)  
        loss = criterion(outputs, labels) 
        loss.backward()  
        optimizer.step() 
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate_model_combined(model, validation_loader, criterion, device):
    model.eval() 
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  
        for data in tqdm.tqdm(validation_loader):
            pixel_values,inputs_vgg, labels = data['inputs']['pixel_values'],data['inputs_vgg'],data["class"]
            pixel_values,inputs_vgg, labels = pixel_values.to(device),inputs_vgg.to(device), labels.to(device)
            
            outputs = model(pixel_values,inputs_vgg) 
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(validation_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc




def run_training_combined(model, train_loader, validation_loader, epochs, device,transformer_epochs=5):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        if epoch==0:
            print("Epoch: 0, Let's freeze everything except for the cross-transformer and classifier")
            model.unfreeze_transformer()
            model.freeze_vgg()
            model.freeze_vitmae()
            model.unfreeze_vgg_head()
        if epoch==transformer_epochs:
            print(f"Epoch: {transformer_epochs}, Let's unfreeze vgg16")
            model.unfreeze_vgg()
        train_loss, train_acc = train_model_combined(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_model_combined(model, validation_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')
        torch.save(model.state_dict(), '../weights/combined.pth')
    return model