import pandas as pd
import numpy as np
import torch
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

X_train_tensor = torch.from_numpy(X_train)
X_test_tensor = torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)

print(X_train_tensor.shape)     # 30 features ~ 30 weights, 1 bias


# Defining the model
class MySimpleNN():
    def __init__(self,X):
        self.weights = torch.rand(X.shape[1],1, dtype=torch.float64, requires_grad=True)
        self.bias = torch.zeros(1,dtype=torch.float64, requires_grad=True)

    def forward_pass(self,X):
        z = torch.matmul(X,self.weights) + self.bias
        y_pred = torch.sigmoid(z)
        return y_pred
    def loss_function(self,y_pred,y):
        # clamp predictions to avoid log(0)
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred,epsilon,1-epsilon)

        # calculate loss
        loss = -(y_train_tensor * torch.log(y_pred) + (1-y_train_tensor) * torch.log(1-y_pred)).mean()
        return loss

# Important Parameters
learning_rate = 0.1
epochs = 25

# training pipeline
#create model
model = MySimpleNN(X_train_tensor)
# print(model.weights)
# print(model.bias)

# define loop
for epoch in range(epochs):

    # forward pass
    y_pred = model.forward_pass(X_train_tensor)
    # print(y_pred)

    # loss calculate
    loss = model.loss_function(y_pred,y_train_tensor)
    # print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    # backward pass
    loss.backward()
    
    # parameters update
    # w_new = w_old - lr(dl/dw)
    with torch.no_grad():
        model.weights -= learning_rate * model.weights.grad
        model.bias -= learning_rate * model.bias.grad

    # zero gradients
    model.weights.grad.zero_()
    model.bias.grad.zero_()

    # print loss in each epoch
    print(f"Epoch: {epoch +1}, Loss: {loss.item()}")

print(model.weights)
print(model.bias)
print("="*50)

# model evaluation
with torch.no_grad():
    y_pred = model.forward_pass(X_test_tensor)
    y_pred = (y_pred > 0.5).float()
# print(y_pred)
    accuracy = (y_pred == y_test_tensor).float().mean()
    print(f'Accuracy: {accuracy.item()}')