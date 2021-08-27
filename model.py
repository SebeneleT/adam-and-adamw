import torch
import torch.nn as nn
import torch.nn.functional as F





#EXPERIMENT : LOGISTIC REGRESSION

class Logistic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0),-1)
        outputs = self.linear(x)
        return outputs



#EXPERIMENT : CONVOLUTIONAL NEURAL NETWORKS

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1)
        #self.pool = nn.MaxPool2d(3, 2)
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(in_features=128, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        #x = self.pool(x)
        x = F.relu(self.conv2(x))
        #x = self.pool(x)
        x = F.relu(self.conv3(x))
        #x = self.pool(x)
        # get the batch size and reshape
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out


 # EXPIRIMENT : MULTI LAYER NEURAL NETWORK
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,10)
        self.droput = nn.Dropout(0.2)
        
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.droput(x)
        x = F.relu(self.fc2(x))
        x = self.droput(x)
        x = self.fc3(x)
        return x          