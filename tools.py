import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm


# 定义标准化函数
def standardize(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std


# 读取并标准化数据
train_data = pd.read_csv(r'E:\Workspace\function_fitting\fitting_data\function2_train.csv')
val_data = pd.read_csv(r'E:\Workspace\function_fitting\fitting_data\function2_val.csv')

train_X = standardize(train_data.iloc[:, :-1].values)
train_y = train_data.iloc[:, -1].values

val_X = standardize(val_data.iloc[:, :-1].values)
val_y = val_data.iloc[:, -1].values

train_X = torch.tensor(train_X, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)
val_X = torch.tensor(val_X, dtype=torch.float32)
val_y = torch.tensor(val_y, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(train_X, train_y)
val_dataset = TensorDataset(val_X, val_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 定义模型
class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5):
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

    def forward(self, x):
        return self.model(x)


model = FullyConnectedNetwork(4, [64, 128, 256, 256, 128, 64], 1, dropout_rate=0.3)

# 定义优化器和学习率调度器
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# 定义损失函数
loss_fn = nn.MSELoss()


# 验证函数
def val(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item() * data.size(0)
    avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss


# 训练函数
def train(model, train_loader, val_loader, optimizer, loss_fn, scheduler, epochs=100):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data, target in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        scheduler.step()

        train_loss /= len(train_loader.dataset)
        val_loss = val(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}')

        save_path = r'E:\Workspace\function_fitting\result'
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{save_path}/best_model.pth')
            print(f"Saved new best model with validation loss: {best_val_loss}")

    return train_losses, val_losses


train_losses, val_losses = train(model, train_loader, val_loader, optimizer, loss_fn, scheduler, epochs=100)

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
