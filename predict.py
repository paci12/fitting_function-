import torch
import numpy as np
import pandas as pd
from models.FullyConnectedNetwork import FullyConnectedNetwork

# 训练数据已经加载并且模型已经训练好了
# 加载训练数据以计算均值和标准差

# 定义标准化函数
def standardize(data, mean, std):
    return (data - mean) / std


train_data = pd.read_csv(r'E:\Workspace\function_fitting\fitting_data\function3_train.csv')

# 计算训练数据的均值和标准差
mean = train_data.iloc[:, :-1].values.mean(axis=0)
std = train_data.iloc[:, :-1].values.std(axis=0)

# 创建模型实例
model = FullyConnectedNetwork(2, [16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16], 1)

# 加载保存的模型参数
model.load_state_dict(torch.load(r'E:\Workspace\function_fitting\result\best_model.pth'))

# 设置模型为评估模式
model.eval()

# 假设新的输入数据为
new_input = np.array([[60.75448519014384, 17.052412368729154]])  # 你需要替换 x, y, z, w 为实际的数值

# 标准化新的输入数据
standardized_input = standardize(new_input, mean, std)
standardized_input = torch.tensor(standardized_input, dtype=torch.float32)

# 使用模型进行预测
with torch.no_grad():
    prediction = model(standardized_input)
    print(prediction)
    print(std)
    print(mean)
    # prediction = prediction * std + mean
    # print("Predicted output:", prediction.numpy())


# 60.75448519014384,17.052412368729154,-70.38889403731014