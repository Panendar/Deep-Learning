# problems in mini batch gradient descent
#   1. No interface for data
#   2. No easy way to apply transformations
#   3. Shuffling and sampling(random,custom samplers)
#   4. Batch management & Parallelization(num_workers)

# To solve them we have two classes Dataset and Dataloader

# Dataset class -> blueprint, where you create a custom dataset , you decide how data is loaded and returned
        # -> __init__() - tells how data is loaded 
        # -> __len__()  - returns the total number od samples
        # -> __getitem__(index)  - returns the data(and label) at given index


# Dataloader class -> wraps the dataset and handles the batching, shuffling and parallel loading for us
# control flow:
    # At the start of each epoch, the DataLoader (if shuffle=True) shuffles indices (using a sampler).
    # It divides the indices into chunks of batch_size.
    # for each index in the chunk, data samples are fetched from the Dataset object
    # The samples are then collected and combined into a batch (using collate_fn)
    # The batch is returned to the main training loop

from sklearn.datasets import make_classification
import torch

# create the synthetic classification dataset using sklearn
X,y = make_classification(
    n_samples=10,       # number of samples
    n_features=2,       # number of features
    n_informative=2,    # number of information features
    n_redundant=0,      # number of redundant features
    n_classes=2,        # number of classes
    random_state=42     # For reproducibility 
)

# print(X)
# print(X.shape)
# print(y)

# convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):

    def __init__(self,features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self,index):
        return self.features[index], self.labels[index]
    
dataset = CustomDataset(X,y)
# print(len(dataset))
# print(dataset[2])

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_features, batch_labels in dataloader:
    print(batch_features)
    print(batch_labels)
    print("-"*50)