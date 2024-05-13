import torch
import torch.nn as nn
import torch.nn.functional as F

# class FullyConnectedNetwork(nn.Module):
#     def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5):
#         super(FullyConnectedNetwork, self).__init__()
#         # 定义第一层
#         self.fc1 = nn.Linear(input_size, hidden_sizes[0])
#         self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
#         # 定义第二层
#         self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
#         self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
#         # 定义第三层
#         self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
#         self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
#         # 定义第四层
#         self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
#         self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
#         # 定义第五层
#         self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
#         self.bn5 = nn.BatchNorm1d(hidden_sizes[4])
#         # 定义第六层
#         self.fc6 = nn.Linear(hidden_sizes[4], hidden_sizes[5])
#         self.bn6 = nn.BatchNorm1d(hidden_sizes[5])
#         # 定义输出层
#         self.fc7 = nn.Linear(hidden_sizes[5], output_size)
#         self.dropout = nn.Dropout(dropout_rate)
#
#     def forward(self, x):
#         x = self.dropout(torch.nn.functional.relu(self.bn1(self.fc1(x))))
#         x = self.dropout(torch.nn.functional.relu(self.bn2(self.fc2(x))))
#         x = self.dropout(torch.nn.functional.relu(self.bn3(self.fc3(x))))
#         x = self.dropout(torch.nn.functional.relu(self.bn4(self.fc4(x))))
#         x = self.dropout(torch.nn.functional.relu(self.bn5(self.fc5(x))))
#         x = self.dropout(torch.nn.functional.relu(self.bn6(self.fc6(x))))
#         x = self.fc7(x)
#         return x
class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(FullyConnectedNetwork, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.model(x)
        # x = self.sigmoid(x)
        return x

