"""Cabezas de TransLowNet (detector MIL + clasificador).

Espejo fiel de `Propuesta2/models_Inference.py` (mismas capas → compatibles con
los .pkl de pesos), pero SIN `torch.set_default_tensor_type('torch.cuda.FloatTensor')`
para que el edge cargue también en CPU (dev) y no solo en Jetson.

Si en el futuro se empaqueta el modelo como `pip -e` (models/translownet), estas
clases se importarán de ahí y este archivo desaparecerá.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.init as torch_init


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Detector_VAD(nn.Module):
    """Detector de anomalía (score [0,1]) — usado con x3d_l."""

    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs):
        x = self.dropout(self.relu(self.fc1(inputs)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.sigmoid(self.fc3(x))


class Model_V3_Connection(nn.Module):
    """Detector con conexiones de atención — usado con x3d_m / x3d_s."""

    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc_att1 = nn.Sequential(nn.Linear(n_features, 512), nn.Softmax(dim=1))
        self.fc2 = nn.Linear(512, 32)
        self.fc_att2 = nn.Sequential(nn.Linear(512, 32), nn.Softmax(dim=1))
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs):
        x = self.fc1(inputs)
        att1 = self.fc_att1(inputs)
        x = (x * att1) + x
        x = self.dropout(self.relu(x))
        att2 = self.fc_att2(x)
        x = self.fc2(x)
        x = (x * att2) + x
        x = self.dropout(self.relu(x))
        return self.sigmoid(self.fc3(x))


class violenceOneCrop(nn.Module):
    """Clasificador de tipo de anomalía (softmax sobre n_classes)."""

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, n_classes)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.apply(weight_init)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.softmax(self.fc3(x))
