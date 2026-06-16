import torch


# y = x**2
# requires grad is used to track the gradients of the tensor during back propagation.
# x = torch.tensor(3.0, requires_grad=True)
# y = x**2
# print(f"{x}, {y}")                    # after this the computational graph will be formed behind which looks like x -> sqrt -> y
# y.backward()
# print(x.grad)   # print's 6 the dy/dx at x = 3 => 2*3=6


# y = x**2, z = sin(y)
# x = torch.tensor(4.0,requires_grad=True)
# y = x**2
# z = torch.sin(y)
# print(f"{x},{y},{z}")           # the computational graph will be formed behind which looks like x -> sqrt -> y -> sin -> z
# z.backward()
# print(x.grad)                   # we can't perform grad on intermediate nodes like y in this case


# neural network 
#  x -> w -> sigmoid -> y_hat/y_pred -> Loss
# z = wx + b
# activation function (σ)
# loss = -[y_target.ln(y_pred)+(1-y_target).ln(1-y_pred)]


# # Manually
 

# we have to find dL/dw and dL/db
# dL/dw = dL/dy_pred * dy_pred/dz * dz/dw similarly for dL/db

# x = torch.tensor(6.7, requires_grad=True)   # Input Feature
# y = torch.tensor(0.0)   # True label(binary)

# w = torch.tensor(1.0)   # Weight
# b = torch.tensor(0.0)  # Bias

# # Binary Cross-Entropy Loss for scalar
# def loss_func(prediction, target):
#     epsilon = 1e-8  # To prevent log(0)
#     prediction = torch.clamp(prediction,epsilon,1-epsilon)
#     return -(target * torch.log(prediction) + (1-target) * torch.log(1-prediction))


# # Forward Pass
# z = w * x + b       # Weighted sum(linear part)
# y_pred = torch.sigmoid(z)   #predicted probability

# # compute binary cross entropy loss
# loss = loss_func(y_pred,y)

# # Derivatives
# # # 1. dL/d(y_pred): Loss with respect to the prediction    (y_pred)
# dloss_dy_pred = (y_pred - y)/(y_pred*(1-y_pred))

# # 2. dy_pred/dz: prediction (y_pred) with respect to z (sigmoid derivative)
# dy_pred_dz = y_pred * (1-y_pred)

# # 3. dz/dw and dz/db: z with respect to w and b
# dz_dw = x
# dz_db = 1

# dL_dw = dloss_dy_pred * dy_pred_dz * dz_dw
# dL_db = dloss_dy_pred * dy_pred_dz * dz_db

# print(dL_dw)
# print(dL_db)



# Using Autograd

# x = torch.tensor(6.7)       # we are not computing gards w.r.t x,y
# y = torch.tensor(0.0)
# w = torch.tensor(1.0,requires_grad=True)
# b = torch.tensor(0.0,requires_grad=True)

# # forward pass
# z = w * x + b
# print(z)
# y_pred = torch.sigmoid(z)
# print(y_pred)
# loss =  loss_func(y_pred,y)

# # backward
# loss.backward()
# print(w.grad)
# print(b.grad)




# vectors

# x = torch.tensor([1.,2.,3.],requires_grad=True)
# y = (x**2).mean()
# y.backward()
# print(x.grad)


# clearing grad
# if we use backward multiple times, the gradient doesn't clear itself and continues from after like x.grad -> 4 -> 8 ... it goes on adding

# x.gard.zero_()          # set's to zero


# disable gradient tracking
# we have trained using backward and now we have to predict the outcome in this we have completed training so we only need forward pass so in this we have to disable back tracking

x = torch.tensor(4.0, requires_grad=True)
y = x**2
y.backward()
print(y)

# option -1 requires_grad_(False)
# print(x.requires_grad_(False))
# print(y)
# we can't call backward
# y.backward => error

# option-2 detach
z = x.detach()
y1 = z **2
# y1.backward()              we can't perform

# option - 3 torch.no_grad():
with torch.no_grad():
    y = x ** 2
    print(y)
# y1.backward()              we can't perform