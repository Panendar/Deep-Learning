# Deep Learning Repository 🧠

A collection of deep learning projects and experiments using PyTorch.

## 📁 Repository Structure

```
Deep Learning/
├── README.md           # Project documentation
├── my_1st-NN.py       # MNIST digit classifier implementation
└── data/              # Dataset storage (auto-generated)
```

## 🚀 Projects

### 1. MNIST Digit Classifier (`my_1st-NN.py`)

A simple feedforward neural network that classifies handwritten digits from the MNIST dataset.

**Architecture:**

- Input Layer: 784 neurons (28×28 pixels)
- Hidden Layer: 128 neurons with ReLU activation
- Output Layer: 10 neurons (digits 0-9)

**Features:**

- ✅ Automated data loading and preprocessing
- ✅ Normalization with mean=0.5, std=0.5
- ✅ Training with Adam optimizer
- ✅ Real-time accuracy tracking
- ✅ Test set evaluation
- ✅ Single image prediction with confidence scores

**Performance:**

- Training Epochs: 5
- Test Accuracy: ~97-98%
- Loss Function: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- torchvision

### Setup

1. Clone or download this repository:

```bash
cd "c:\Users\venkata\Downloads\Gen_AI\Deep Learning"
```

2. Install dependencies:

```bash
pip install torch torchvision
```

Or using conda:

```bash
conda install pytorch torchvision cpuonly -c pytorch
```

## 💻 Usage

### Running the MNIST Classifier

```bash
python my_1st-NN.py
```

**What happens:**

1. Downloads MNIST dataset (first run only)
2. Trains the model for 5 epochs
3. Displays training loss and accuracy per epoch
4. Evaluates on test set
5. Prompts for an image index to predict

**Example Output:**

```
Epoch 1/5, Loss: 0.2631, Accuracy: 92.21%
Epoch 2/5, Loss: 0.1170, Accuracy: 96.52%
Epoch 3/5, Loss: 0.0804, Accuracy: 97.50%
Epoch 4/5, Loss: 0.0617, Accuracy: 98.05%
Epoch 5/5, Loss: 0.0475, Accuracy: 98.43%
Test Accuracy: 97.50%

Enter image index (0-9999) from test dataset: 42
True Label: 3, Predicted: 3, Confidence: 0.99
```

## 📊 Model Details

### Network Architecture

```
SimpleNN(
  (flatten): Flatten()
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
```

### Hyperparameters

| Parameter     | Value            |
| ------------- | ---------------- |
| Batch Size    | 64               |
| Learning Rate | 0.001            |
| Epochs        | 5                |
| Optimizer     | Adam             |
| Loss Function | CrossEntropyLoss |

### Data Preprocessing

- Convert to tensor
- Normalize: `(pixel - 0.5) / 0.5` → range [-1, 1]

## 🧪 Experiments & Results

### MNIST Classification

- **Dataset**: 60,000 training images, 10,000 test images
- **Input**: 28×28 grayscale images
- **Classes**: 10 digits (0-9)
- **Achieved Accuracy**: ~97-98% on test set

## 📚 Learning Resources

### Concepts Covered

- [x] Data loading and preprocessing
- [x] Neural network architecture design
- [x] Forward propagation
- [x] Backpropagation
- [x] Gradient descent optimization
- [x] Loss functions (CrossEntropyLoss)
- [x] Model evaluation
- [x] Inference and predictions

## 🔧 Troubleshooting

### Common Issues

**1. Module not found error:**

```bash
pip install torch torchvision
```

**2. CUDA out of memory (if using GPU):**

- Reduce batch size in the code
- Or use CPU by setting `device = torch.device("cpu")`

**3. Dataset download fails:**

- Check internet connection
- Dataset auto-downloads to `./data/` folder

## 🎯 Future Enhancements

- [ ] Add more advanced architectures (CNN, RNN, Transformer)
- [ ] Implement data augmentation
- [ ] Add model checkpointing
- [ ] Create visualization tools for predictions
- [ ] Add TensorBoard logging
- [ ] Implement transfer learning examples
- [ ] Add more datasets (CIFAR-10, Fashion-MNIST)

## 📝 Notes

- The model automatically uses GPU if available, otherwise falls back to CPU
- Data is cached after first download in `./data/` directory
- Each run will show different random initialization results

## 🤝 Contributing

Feel free to experiment with:

- Different network architectures
- Hyperparameter tuning
- Additional datasets
- Advanced training techniques

## 📄 License

This project is for educational purposes.

---

**Author**: Venkata  
**Last Updated**: October 24, 2025  
**PyTorch Version**: 2.9.0+cpu
