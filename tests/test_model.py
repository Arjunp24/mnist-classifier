import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from mnist_classifier import SmallCNN, count_parameters

@pytest.fixture
def model():
    return SmallCNN()

def test_parameter_count(model):
    param_count = count_parameters(model)
    assert param_count < 20000, f"Model has {param_count} parameters, which exceeds 20,000"

def test_batch_normalization_exists(model):
    has_bn = False
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            has_bn = True
            break
    assert has_bn, "Model does not use batch normalization"

def test_dropout_exists(model):
    has_dropout = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            has_dropout = True
            break
    assert has_dropout, "Model does not use dropout"

def test_fully_connected_exists(model):
    has_fc = False
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            has_fc = True
            break
    assert has_fc, "Model does not use fully connected layers"

def test_model_accuracy():
    # Load the saved model and test accuracy
    # You'll need to save model accuracy during training
    accuracy_file = "test_accuracy.txt"
    assert os.path.exists(accuracy_file), "Accuracy file not found"
    
    with open(accuracy_file, 'r') as f:
        accuracy = float(f.read().strip())
    
    assert accuracy >= 99.4, f"Model accuracy {accuracy}% is below 99.4%" 