import torch
print(torch.__version__)

# Check if GPU is Available
# if torch.cuda.is_available():
#     print("GPU Available!")
#     print(f"Using GPU {torch.cuda.get_device_name(0)}")
# else:
#     print("GPU Not Available")
# print("="*50)

# # Creating the Tensors

# # Empty function
# a = torch.empty(2,3)          # this creates the empty tensor and no values are assigned but it sometimes returns the values present at the memory in colab before
# print(f"{a},\n type:{type(a)}")
# print("="*50)

# # Zeros function
# b = torch.zeros(2,3)
# print(b)
# print("="*50)

# # Ones Function
# c = torch.ones(2,3)
# print(c)
# print("="*50)

# # Random function
# d = torch.rand(2,3)
# print(d)

# # use of seed
# e = torch.rand(2,3)             # if we observe every time we create an random the values are different so to avoid that we use manual seed 
# print(e)
# print("="*50)
# # Manual seed
# f = torch.manual_seed(100)
# print(torch.rand(2,3, generator=f))

# g = torch.manual_seed(100)
# print(torch.rand(2,3, generator=g))
# print("="*50)

# using tensor (custom- tensors)
# h  = torch.tensor([[1,2,3],[4,5,6]])
# print(h)

# print(f"Using arange -> {torch.arange(0,10,2)}")
# print(f"Using Linspace -> {torch.linspace(0,10,10)}")
# print(f"Using eys -> {torch.eye(5)}")
# print("Using Full ->", torch.full((3,3),5))

# Tensor Shape
# print(h.shape)

# # to create the same shape tensor 
# # print("Using Empty like:",torch.empty_like(h))
# # print("Using Zeros like:",torch.zeros_like(h))
# # print("Using Ones like:",torch.ones_like(h))

# # Tensor Data Type
# print(h.dtype)
# print(torch.tensor([1.0,2.0,3.0],dtype=torch.int32))

# # change to another data type
# print(h.to(torch.float32))
# print("="*50)

# # using rand_like
# print(torch.rand_like(h,dtype=torch.float32))



# Scalar Operations

# a = torch.rand(2,2)
# print(a)
# # addition
# print(a + 2)
# # subtraction
# print(a -2)
# # multiplication
# print(a * 3)
# # division
# print(a / 3)
# # int division
# print(a * 100 // 3)
# # mod 
# print((a * 100 // 3)%2)
# # power
# print(a**2)

# Element wise operations

a = torch.rand(2,3)
# b = torch.rand(2,3)

# print(a)
# print(b)
# print(a + b)
# print(a - b)
# print(a * b)
# print(a / b)
# print(a // b)
# print(a ** b)
# print(a % b)


# c = torch.tensor([1,-2,3,-4])

# # absolute value
# print(torch.abs(c))
# # negative 
# print(torch.neg(c))
# #round
# print(a)
# print(torch.round(a))
# # ceil
# print(torch.ceil(a))
# # floor
# print(torch.floor(a))
# # clamp
# print(torch.clamp(c,min=1, max=5))


# Reduction operations
# e = torch.randint(low=0,high=10, size=(2,3))
# print(e)

# # Sum
# print(torch.sum(e))
# # sum by columns
# print(torch.sum(e,dim=0))
# # Row sum
# print(torch.sum(e,dim=1))

# # mean, standard deviation, variance only works for float and complex dtypes
# f = torch.rand(2,3)
# print(torch.mean(f))
# # column mean
# print(torch.mean(f,dim=0))

# # median
# print(torch.median(e))

# # min and max
# print(torch.min(e))
# print(torch.max(e))

# # product
# print(torch.prod(e))

# # standard deviation
# print(torch.std(f))

# # variance
# print(torch.var(f))

# # argmax/ argmin - position of the max/min element
# print(torch.argmax(e))
# print(torch.argmin(e))

# matrix multiplication

# a = torch.randint(low=0,high=10,size=(2,3))
# b = torch.randint(low=0,high=20,size=(3,2))
# print(torch.matmul(a,b))

# # vector - dot product
# c = torch.tensor([1,2,3])
# d = torch.tensor([5,4,8])
# print(torch.dot(c,d))

# # transpose # mention which dimension with which
# print(torch.transpose(a,0,1))

# e = torch.randint(size=(3,3),low=0,high=10,dtype=torch.float32)
# # determinant
# print(torch.det(e))

# # inverse
# print(torch.inverse(e))


#  Comparison matrix
# a = torch.randint(low=0,high=10,size=(2,3))
# b = torch.randint(low=0,high=10,size=(2,3))

# print(a>b)
# print(a<b)
# print(a==b)
# print(a!=b)
# print(a>=b)
# print(a<=b)


# Special functions
# a = torch.randint(low=0,high=10,size=(2,3), dtype=torch.float32)

# # log 
# print(torch.log(a))

# # exponential
# print(torch.exp(a))

# # square root
# print(torch.sqrt(a))

# # sigmoid
# print(torch.sigmoid(a))

# # softmax
# print(torch.softmax(a,dim=0))

# # relu
# print(torch.relu(a))

# Inplace operations used when we have a large dataset since the operation creates a new memory so to reduce that we use these 
# a = torch.randint(low=0,high=10,size=(2,3), dtype=torch.float32)
# b = torch.randint(low=0,high=10,size=(2,3), dtype=torch.float32)
# print(a.add_(b))
# print(a)
# print(a.relu_())


# Copying the tensors

# a = torch.rand(2,3)
# # b = a                   # the problem is if we change the a then the b will also change
# # print(a)
# # a[0][0] = 10
# # print(a)
# # # memory of a and b is same by using this 
# # print(id(a))
# # print(id(b))

# # so we use clone to avoid these
# c = a.clone()
# print(id(a))
# print(id(c))
# a[1][0]= 100
# print(a)
# print(c)


# Tensor operations on GPU

# torch.cuda.is_available()
# device = torch.device('cuda')

# # creating the new tensor on GPU
# torch.rand((2,3),device=device)

# # moving the existing tensors
# a = torch.rand(2,3)
# b = a.to(device)



# Reshaping Tensors

# a = torch.ones(4,4)
# print(a)

# # reshape
# b =a.reshape(2,2,2,2)
# print(b)

# # flatten
# c = b.flatten()
# print(c)

# # permute
# d = torch.rand(2,3,4)
# print(d.shape)
# print(d.permute(2,0,1).shape)

# unsqueeze
# e = torch.rand(256,256,3)
# print(e.unsqueeze(0).shape)
# print(e.unsqueeze(1).shape)

# # squeeze
# print(e.squeeze(1).shape)



# NumPy and PyTorch
import numpy as np

a = torch.rand(2,3)
b = a.numpy()
print(f"{b},{type(b)}")

c = np.array([1,2,3])
d = torch.from_numpy(c)
print(f"{d},{type(d)}")