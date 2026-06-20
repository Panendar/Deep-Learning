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


from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):

    def __init__(self,features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self,idx):
        return self.features[idx], self.labels[idx]
    
train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor)

print(train_dataset[:5])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# defining the model    

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


# creating model
model = MySimpleNN(X_train_tensor.shape[1])

# define optimizer
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
# model.parameters iterate over all your trainable weights and bias and adjust's them

# defining the loss function
loss_function = nn.BCELoss()

# define loop
for epoch in range(epochs):

    for batch_features, batch_labels in train_loader:

        # Here we are using batch gradient descent, so we are using the whole training data in one go, it has two problems 1-> memory inefficiency 2-> Better convergence, so we will use mini batch gradient descent instead 
        y_pred = model(batch_features)

        # loss calculation
        loss = loss_function(y_pred,batch_labels.view(-1,1))          # keep in mind to match the shape use view instead of reshape function to change

        # clear gradients -> back to zero
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # parameter updates
        optimizer.step()

        # print loss in each epoch
        print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')


# model evaluation
model.eval()        # set the model to evaluation mode
accuracy = []
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        y_pred = model(batch_features)
        y_pred = (y_pred > 0.8).float()     # convert probabilities to binary predictions
        
        # calculate accuracy for the current batch
        batch_accuracy = (y_pred.view(-1) == batch_labels).float().mean()
        accuracy.append(batch_accuracy)

# calculate overall accuracy
overall_accuracy = sum(accuracy)/ len(accuracy)
print(f"Accuracy: {overall_accuracy:.4f}")