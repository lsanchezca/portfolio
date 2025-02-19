
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Red neuronal MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Aplanar la entrada
        x = self.hidden_layer1(x)
        x = self.activation(x)
        x = self.hidden_layer2(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x


# Ejemplo de uso
input_size = 3 * 25 * 25  # Tamaño de la entrada para imágenes 25x25 con 3 canales
hidden_size = 512  # Número de neuronas en la capa oculta (puedes ajustarlo)
output_size = 43  # Número de clases para la salida (ajústalo si es diferente)

# Crear una instancia de la red
mlp = MLP(input_size, hidden_size, output_size)

# Crear un tensor de ejemplo con un batch de tamaño 16 y 25x25 píxeles de entrada
example_input = torch.randn(16, 3, 25, 25)  # Tamaño (batch_size, 3, 25, 25)

# Realizar una pasada hacia adelante
output = mlp(example_input)

print("Tamaño de la salida:", output.shape)  # Debería ser [16, 43] (16 imágenes, 43 clases)


# Red neuronal CNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()

        # Primer bloque convolucional
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Segundo bloque convolucional
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Tercer bloque convolucional
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Pooling global
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Capa totalmente conectada
        self.fc = nn.Linear(in_features=64, out_features=output_dim)

    def forward(self, x):
        # Primer bloque convolucional
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Segundo bloque convolucional
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Tercer bloque convolucional
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Pooling global
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Capa totalmente conectada
        x = self.fc(x)

        # Softmax para clasificación
        x = F.log_softmax(x, dim=1)

        return x

  