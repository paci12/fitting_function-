import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from models.FullyConnectedNetwork import FullyConnectedNetwork
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 设置随机种子

# Load test data
test_data = pd.read_csv(r'E:\Workspace\function_fitting\fitting_data\function6_test.csv')

# 提取输入和输出
test_X = test_data.iloc[:, :-1].values
test_y = test_data.iloc[:, -1].values  # 所有行，只取最后一列

test_X = torch.tensor(test_X, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32).unsqueeze(1)

test_dataset = TensorDataset(test_X, test_y)

# Create DataLoader
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

# Create model
model = FullyConnectedNetwork(8, [70, 70,70, 70], 1)

# Load the model state dictionary
state_dict = torch.load(r'E:\Workspace\function_fitting\result\best_model.pth')
model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode
loss_fn = torch.nn.MSELoss()  # 用于回归任务的均方误差损失

# Collect predictions and calculate loss
input_output_pairs = []
total_loss = 0.0
num_batches = 0

with torch.no_grad():
    for inputs, target in test_loader:
        outputs = model(inputs)
        loss = loss_fn(outputs, target)
        total_loss += loss.item()
        input_output_pairs.extend(zip(inputs.numpy(), outputs.numpy(), target.numpy()))
        num_batches += 1

# Calculate average loss
average_loss = total_loss / num_batches
print(f'Average Loss: {average_loss:.4f}')

# Convert input-output pairs to numpy arrays
inputs_np, predictions_np, targets_np = zip(*input_output_pairs)
inputs_np = np.array(inputs_np)
predictions_np = np.array(predictions_np).flatten()
targets_np = np.array(targets_np).flatten()

# Sort based on the first input variable for consistent plotting
sorted_indices = np.argsort(inputs_np[:, 0])
sorted_inputs = inputs_np[sorted_indices]
sorted_predictions = predictions_np[sorted_indices]
sorted_targets = targets_np[sorted_indices]

# Plot real data and predicted data
plt.scatter(sorted_inputs[:, 7], sorted_targets, color='blue', label='Real Data')
plt.scatter(sorted_inputs[:, 7], sorted_predictions, color='red', label='Predictions', alpha=0.5)

# Add legend and labels
plt.legend()
plt.title('Comparison of Real and Predicted Data')
plt.xlabel('First Input Variable')
plt.ylabel('Result')

# Show plot
plt.show()
