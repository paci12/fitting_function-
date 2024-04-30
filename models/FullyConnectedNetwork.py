import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FullyConnectedNetwork, self).__init__()
        # 定义第一层
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        # 定义第二层
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        # 定义第三层
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        # 定义第四层
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        # 定义输出层
        self.fc5 = nn.Linear(hidden_sizes[3], output_size)
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x