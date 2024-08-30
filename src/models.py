import torch
import torch.nn as nn

class LegoNet(nn.Module):
    def __init__(self):
        super(LegoNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(1 * 14 * 14, 8)  
        self.fc2 = nn.Linear(8, 2)             
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  

        # for GradNorm:
        self.loss_weights = nn.Parameter(torch.tensor([1., 1.]).float())
        self.last_shared_layer = self.fc2

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(x)  
        x = x.view(-1, 1 * 14 * 14)  
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.tanh(x)
        return x
