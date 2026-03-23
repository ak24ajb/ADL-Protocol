import torch
import torch.nn as nn
import torch.nn.functional as F


#Inheriting 'nn.Module' which is PyTorch's base class for all neural networks

class MNISTNet(nn.Module):
    
    #Setup of the NN
    def __init__(self):
        
        #Declaring super here to intialize the parent class
        super(Net, self).__init__()

        #Layer 1 of the N.network
        #28x28 pixels = 784 flattened inputs, mapped onto 128 neurons
        self.fc1 = nn.Linear(784, 128)

        #Layer 2
        #128 values now mapped to 64 neurons
        self.fc2 = nn.Linear(64, 128)

        #OUTPUT LAYER
        #64 values now mapped to 10 neurons, one for each digit (0-9)
        self.fc3 = nn.Linear(64, 10)


    #how data flows through the NN
    #feedforward neural network
    def forward(self, x):

        #Flattens the image into 784 values, -1 to auto figure out the batch size 
        x = x.view(-1, 784)

        #Applying RELU activation after layer 1/2 passage, so that any negative numbers are clipped to 0
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)
