import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv')
# print(df.head())
# print(df.shape)

# the two columns ID, unnamed:32 are not useful so we remove them
df.drop(columns=['id','Unnamed: 32'], inplace = True)
# print(df.head())

# train test split
X_train,X_test, y_train, y_test = train_test_split(df.iloc[:,1:],df.iloc[:,0],test_size=0.2)

# scaling
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
# print(X_train)
# print(y_train)

# neural net can't understand the letters in y_train so we encode them for NN

# label encoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
# print(y_train)

# Numpy arrays to pytorch tensors

X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train).float()
y_test_tensor = torch.from_numpy(y_test).float()

print(X_train_tensor.shape)     # 30 features ~ 30 weights, 1 bias


class MySimpleNN(nn.Module):
    
    def __init__(self,num_features):

        super().__init__()
        self.linear = nn.Linear(num_features,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,features):
        out = self.linear(features)
        out = self.sigmoid(out)
        return out


learning_rate = 0.1
epochs = 25

loss_function = nn.BCELoss()


# creating model
model = MySimpleNN(X_train_tensor.shape[1])

# define optimizer
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
# model.parameters iterate over all your trainable weights and bias and adjust's them

# define loop
for epoch in range(epochs):

    # forward pass
    y_pred = model(X_train_tensor)

    # loss calculation
    loss = loss_function(y_pred,y_train_tensor.view(-1,1))          # keep in mind to match the shape use view instead of reshape function to change

    # clear gradients -> back to zero
    optimizer.zero_grad()

    # loss backward
    loss.backward()

    # parameter updates
    optimizer.step()

    # print loss in each epoch
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')


# model evaluation
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred = (y_pred > 0.5).float()
# print(y_pred)
    accuracy = (y_pred == y_test_tensor.view(-1,1)).float().mean()
    print(f'Accuracy: {accuracy.item()}')