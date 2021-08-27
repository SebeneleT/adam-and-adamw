import torch
import torch.nn as nn
import torchvision

from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import model
import Adam
import AdamW
import torch.optim as optim
import torch.nn as nn




# learning parameters
input_size = 28*28
#hidden dims for nn
hidden = 128
num_classes = 10
batch_size = 128
epochs = 50
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # define transforms
# transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             #transforms.Resize((28,28)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,)), #mnist
#             #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#cifar
#         ])
# transform_val = transforms.Compose([
#     #transforms.Resize((28,28)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,)), #mnist
#     #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#cifar
# ])

# # train and validation data
# train_data = datasets.CIFAR10(
#     root='../input/data',
#     train=True,
#     download=True,
#     transform=transform_train
# )
# val_data = datasets.CIFAR10(
#     root='../input/data',
#     train=False,
#     download=True,
#     transform=transform_val
# )
# # training and validation data loaders
# train_loader = DataLoader(
#     train_data,
#     batch_size=batch_size,
#     shuffle=True
# )
# val_loader = DataLoader(
#     val_data,
#     batch_size=batch_size,
#     shuffle=False
#)


#when using MNIST
train_data = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

val_data = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                           batch_size=batch_size, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_data, 
                                          batch_size=batch_size, 
                                          shuffle=False)




#Uncomment and choose model to use
#model = Net().to(device)
model = Logistic(input_dim=input_size,output_dim=num_classes).to(device)
#model = CNN().to(device)




criterion = nn.CrossEntropyLoss()


def fit(model, dataloader):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        data.size()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/len(dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(dataloader.dataset)    
    return train_loss, train_accuracy


#validation function
def validate(model, dataloader):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()
        
        val_loss = val_running_loss/len(dataloader.dataset)
        val_accuracy = 100. * val_running_correct/len(dataloader.dataset)        
        return val_loss, val_accuracy

# uncomment to determine which optimiser to use
optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
#optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
#optimizer = optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9)


train_loss  = []
train_accuracy = []
val_loss = []
val_accuracy = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_accuracy = fit(model, train_loader)
    val_epoch_loss, val_epoch_accuracy = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
    print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')   



     