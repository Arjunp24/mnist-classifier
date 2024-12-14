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

def test_fully_connected_or_gap_exists(model):
    has_fc_or_gap = False
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.AvgPool2d, torch.nn.AdaptiveAvgPool2d)):
            has_fc_or_gap = True
            break
    assert has_fc_or_gap, "Model does not use either fully connected layers or global average pooling"