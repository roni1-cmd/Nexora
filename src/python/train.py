import torch
import torch.nn as nn

class NexoraModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=1):
        super(NexoraModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x
