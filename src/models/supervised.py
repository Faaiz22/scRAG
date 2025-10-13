"""
scRAG/src/models/supervised.py

Tiny PyTorch classifier example for scRNA-seq (dense MLP).
"""
import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512,128], n_classes=10, dropout=0.2):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
