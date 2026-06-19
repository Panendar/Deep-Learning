# Building the neural network using nn module
# using the built-in activation function
# using the built-in loss function
# using the built-in optimizer

# key components of the NN module
#   -> Modules(Layers): base class for all neural networks, includes common layers like nn.linear(fully connected layer), nn.Conv2d(Convolution Layer), nn.LSTM(recurrent layer) etc.
#   -> Activation functions: nn.ReLU, nn.sigmoid, and nn.Tanh introduces non-linearity to models
#   -> Loss Function: provides loss functions: nn.CrossEntropyLoss, nn.MSELoss and nn.LLLoss to quantify the difference between the prediction's and targets
#   -> Container Modules: nn.Sequential- a sequential container to stack layers in order
#   -> Regularization and Dropout: layers like nn.Dropout and nn.   BatchNorm2d help prevent overfitting and improve the models ability to generalize to new data


# Create the model class
import torch
import torch.nn as nn
import torchinfo

from torchinfo import summary

# class Model(nn.Module):     # inheriting from the base class nn.module
    
#     def __init__(self,num_features):
        
#         super().__init__()
#         self.linear = nn.Linear(num_features,1)         # a single layer nn with one output
#         self.sigmoid = nn.Sigmoid()

#     def forward(self,features):

#         out = self.linear(features)
#         out = self.sigmoid(out)

#         return out
    
# # creating the dataset
# features = torch.rand(10,5)

# # create model
# model = Model(features.shape[1])

# # forward pass pytorch has a build in __call__ so no need for model.forward(features)
# print(model(features))

# # show weights
# print(model.linear.weight)

# print(summary(model,input_size=(10,5)))     # total 6 trainable parameters 5-inputs + 1 bias


#  creating the nn with 5 input layers, 1-hidden layer with 3 neurons + 1 output layer
#               5*3 = 15 weights for hidden layer and 3*1 =3 weights for output    + 1 bias

class Model_2(nn.Module):
    
    def __init__(self,num_features):

        super().__init__()
        # alternative for the forward pass instead of defining separately use nn.Sequential container
    
        self.network = nn.Sequential(
            nn.Linear(num_features,3),
            nn.ReLU(),
            nn.Linear(3,1),
            nn.Sigmoid()
        )
        # self.linear_1 = nn.Linear(num_features,3)
        # self.relu = nn.ReLU()
        # self.linear_2 = nn.Linear(3,1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self,features):
        # out = self.linear_1(features)
        # out = self.relu(out)
        # out = self.linear_2(out)
        # out = self.sigmoid(out)

        out = self.network(features)

        return out

features = torch.rand(10,5)
model = Model_2(features.shape[1])
print(model(features))
# print(model.linear_1.weight)
print(model.network[0].weight)
# print(model.linear_2.weight)
print(model.network[2].weight)
print(summary(model,input_size=(10,5)))