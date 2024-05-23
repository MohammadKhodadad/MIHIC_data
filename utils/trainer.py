import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
def train_model_classification(model, train_loader, optimizer, criterion, device):
    model.train() 
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data in train_loader:
        inputs, labels = data['image'],data["class"]
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  
        
        outputs = model(inputs)  
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


def validate_model_classification(model, validation_loader, criterion, device):
    model.eval() 
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  
        for data in validation_loader:
            inputs, labels = data['image'],data["class"]
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(validation_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def run_training_classification(model, train_loader, validation_loader, epochs, device):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_model(model, validation_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')
    
    return model