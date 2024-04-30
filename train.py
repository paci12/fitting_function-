import torch.optim as optim
import torch
from models.FullyConnectedNetwork import FullyConnectedNetwork
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

#-------------------定义数据----------------------------------#

train_data = pd.read_csv(r'E:\Workspace\function_fitting\fitting_data\function1_train.csv')
val_data = pd.read_csv(r'E:\Workspace\function_fitting\fitting_data\function1_val.csv')

# 分割特征和目标
train_X = train_data.iloc[:, :-1].values  # 所有行，除最后一列之外的所有列
train_y = train_data.iloc[:, -1].values   # 所有行，只取最后一列

val_X = val_data.iloc[1:, :-1].values # 所有行，除最后一列之外的所有列
val_y = val_data.iloc[1:, -1].values  # 所有行，只取最后一列

train_X = torch.tensor(train_X, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
val_X = torch.tensor(val_X, dtype=torch.float32)
val_y = torch.tensor(val_y, dtype=torch.float32)
print(train_y.shape)
print('****************')
train_dataset = TensorDataset(train_X, train_y)
val_dataset = TensorDataset(val_X, val_y)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

#-------------------定义模型和优化器----------------------------------#
model = FullyConnectedNetwork(4, [32, 128, 128, 32], 1)
# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

#-------------------定义损失函数----------------------------------#
loss_fn = torch.nn.MSELoss()  # 用于回归任务的均方误差损失



def val(model, val_loader, loss_fn):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    total_count = 0
    with torch.no_grad():  # 关闭梯度计算
        for data, target in val_loader:
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item() * data.size(0)
            total_count += data.size(0)
    avg_loss = total_loss / total_count
    return avg_loss

def train(model, train_loader, val_loader, optimizer, loss_fn, epochs=10):
    for epoch in range(epochs):
        best_val_loss = float('inf')
        model.train()  # 确保模型处于训练模式
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
        
        # 在每个 epoch 结束后验证
        val_loss = val(model, val_loader, loss_fn)
        print(f'Epoch {epoch}, Validation Loss: {val_loss}')

        save_path = r'E:\Workspace\function_fitting\result'
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{save_path}/best_model.pth')
            print(f"Saved new best model with validation loss: {best_val_loss}")


train(model, train_loader,val_loader, optimizer, loss_fn,epochs=1000)