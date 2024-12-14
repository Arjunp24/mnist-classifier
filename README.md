# MNIST CNN Classifier

A CNN-based image classifier for the MNIST dataset with specific architectural constraints and performance requirements.

## Project Structure 
```
mnist-classifier
│   README.md # Project documentation
│   requirements.txt # Project dependencies
│   mnist_classifier.py # Main model and training code
|   .gitignore # Git ignore rules
|   test_accuracy.txt # Final model test accuracy
└───.github # GitHub Actions
│   └───workflows
│       │   model_tests.yml 
└───tests/ # Test files
    └───test_model.py # Model architecture tests
```


## Model Requirements
- Parameters < 20,000
- Epochs < 20
- Batch Normalization
- Dropout
- Fully Connected Layer/Global Average Pooling
- Test Accuracy ≥ 99.4%

## Installation

1. Create and activate virtual environment:

Windows
```
python -m venv venv
venv\Scripts\activate
```
Linux/Mac
```
python -m venv venv
source venv/bin/activate
```

2. Install requirements:

```
pip install -r requirements.txt
```

## Training

To train the model:
```
python mnist_classifier.py
```

This will:
- Download MNIST dataset
- Train for 20 epochs
- Save model weights as 'final_model.pth'
- Save test accuracy in 'test_accuracy.txt'

## Testing

Run architecture tests:
```
pytest tests/test_model.py -v
```


Tests verify:
- Parameter count < 20k
- Presence of batch normalization
- Presence of dropout
- Presence of GAP or fully connected layer

## Training Logs

Total trainable parameters: 18596

Epoch 1/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:56<00:00,  8.24it/s, loss=0.228, acc=93.85%] 

Test set: Average loss: 0.0646, Accuracy: 98.06%

Epoch 2/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:59<00:00,  7.91it/s, loss=0.072, acc=97.79%] 

Test set: Average loss: 0.0492, Accuracy: 98.39%

Epoch 3/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:58<00:00,  7.98it/s, loss=0.068, acc=98.02%] 

Test set: Average loss: 0.0572, Accuracy: 98.27%

Epoch 4/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:59<00:00,  7.85it/s, loss=0.064, acc=98.07%] 

Test set: Average loss: 0.0583, Accuracy: 98.07%

Epoch 5/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:01<00:00,  7.65it/s, loss=0.063, acc=98.07%] 

Test set: Average loss: 0.0423, Accuracy: 98.69%

Epoch 6/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:02<00:00,  7.52it/s, loss=0.062, acc=98.11%] 

Test set: Average loss: 0.0411, Accuracy: 98.61%

Epoch 7/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:00<00:00,  7.79it/s, loss=0.059, acc=98.20%] 

Test set: Average loss: 0.0372, Accuracy: 98.68%

Epoch 8/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:00<00:00,  7.71it/s, loss=0.062, acc=98.12%] 

Test set: Average loss: 0.0467, Accuracy: 98.58%

Epoch 00008: reducing learning rate of group 0 to 1.0000e-03.

Epoch 9/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:01<00:00,  7.67it/s, loss=0.035, acc=98.91%] 

Test set: Average loss: 0.0206, Accuracy: 99.34%

Epoch 10/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:03<00:00,  7.44it/s, loss=0.030, acc=99.05%] 

Test set: Average loss: 0.0208, Accuracy: 99.33%

Epoch 11/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:01<00:00,  7.66it/s, loss=0.029, acc=99.11%] 

Test set: Average loss: 0.0201, Accuracy: 99.37%

Epoch 12/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:01<00:00,  7.57it/s, loss=0.027, acc=99.16%] 

Test set: Average loss: 0.0180, Accuracy: 99.35%

Epoch 13/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:01<00:00,  7.68it/s, loss=0.027, acc=99.22%] 

Test set: Average loss: 0.0186, Accuracy: 99.37%

Epoch 14/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:01<00:00,  7.60it/s, loss=0.027, acc=99.17%] 

Test set: Average loss: 0.0193, Accuracy: 99.38%

Epoch 15/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:03<00:00,  7.37it/s, loss=0.027, acc=99.18%] 

Test set: Average loss: 0.0194, Accuracy: 99.41%

Epoch 16/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:05<00:00,  7.19it/s, loss=0.026, acc=99.16%] 

Test set: Average loss: 0.0183, Accuracy: 99.46%

Epoch 17/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:14<00:00,  6.29it/s, loss=0.027, acc=99.19%] 

Test set: Average loss: 0.0169, Accuracy: 99.40%

Epoch 18/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:14<00:00,  6.27it/s, loss=0.025, acc=99.26%] 

Test set: Average loss: 0.0190, Accuracy: 99.44%

Epoch 19/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:22<00:00,  5.69it/s, loss=0.025, acc=99.22%] 

Test set: Average loss: 0.0196, Accuracy: 99.34%

Epoch 00019: reducing learning rate of group 0 to 1.0000e-04.

Epoch 20/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [01:25<00:00,  5.49it/s, loss=0.021, acc=99.36%] 

Test set: Average loss: 0.0163, Accuracy: 99.41%

Training completed!


## Model Architecture

The CNN architecture includes:
- Multiple convolutional layers
- Batch normalization after convolutions
- Dropout regularization (p=0.1)
- Global average pooling
- No fully connected layers