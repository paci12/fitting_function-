import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from models.FullyConnectedNetwork import FullyConnectedNetwork

# 生成训练数据用的数据
x = torch.linspace(-torch.pi, torch.pi, 10000)  # (1000, )
x = torch.unsqueeze(input=x, dim=1)  # (1000, 1)
y = torch.sin(x)  # (1000, 1)

plt.plot(x.numpy(), y.numpy())

# 如果可以用cuda就在cuda上运行，这样会快很多
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device to train")

x = x.to(device)
y = y.to(device)


# 定义NN模型，继承自nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self):
        # 调用
        super(NeuralNetwork, self).__init__()
        #
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 70),
            nn.Sigmoid(),
            nn.Linear(70, 1)
        )

    def forward(self, x):
        y_pred = self.linear_relu_stack(x)
        return y_pred


# 把模型放到GPU上训练
model = FullyConnectedNetwork(1,[70,70,70],1).to(device)
# 均方差做损失函数
loss_fn = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# 用下面这个Adam优化器会收敛的快很多
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 迭代3000次
batches = 3000

plt.figure("regression")  # 新建一张画布，打印数据点和预测值
plt.ion()  # 开启交互模式
plt.show()
for i in range(batches):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    #
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        loss, batch = loss.item(), i
        print(f'loss: {loss} {batch}')
        plt.cla()
        plt.plot(x.cpu().numpy(), y.cpu().numpy())
        plt.plot(x.cpu().numpy(), y_pred.detach().cpu().numpy())
        plt.pause(0.001)
# 保存
torch.save(model.state_dict(), "result/model.pth")
print("Saved PyTorch Model State to model.pth")
