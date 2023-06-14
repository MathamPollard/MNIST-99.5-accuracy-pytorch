import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchsummary import summary
#用GPU训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 设置随机种子
torch.manual_seed(0)

# 定义超参数
batch_size = 512
learning_rate = 0.001
momentum = 0.9
epochs = 25

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#下载到本地，下载到内存，从本地读取，报错

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding = 1)#尺寸不变
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)                                              #尺寸减半
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128*7*7, 512)    #前一层通道数乘以前一层特征图大小
        self.dp1 = nn.Dropout(0.8)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.pool1(x)

        x = self.conv2(x)
        #x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.pool1(x)
        
        x = self.conv3(x)
        #x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.1)

        x = x.view(-1, 128*7*7)
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dp1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
# 实例化模型和优化器
model = Net()
model = model.to(device)  #避免报错
summary(model, (1, 28, 28))
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 训练模型
for epoch in range(1, epochs + 1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)    #避免报错
        target = target.to(device)  #避免报错
        optimizer.zero_grad()    #梯度值是累加的，因此在每一次迭代前需要将梯度清零
        output = model(data)
        loss = criterion(output, target)
        loss.backward()    #该函数用于计算所有可训练参数的梯度，即损失函数对每个参数的导数
        optimizer.step()    #用于更新所有可训练参数
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    # 测试模型
    model.eval()  #评估模式，不更新参数
    test_loss = 0
    correct = 0
    with torch.no_grad(): #禁用梯度计算，减少内存的消耗，加速计算
        for data, target in test_loader:
            data = data.to(device)  #避免报错
            target = target.to(device)  #避免报错
            output = model(data)
            # 累计损失
            test_loss += criterion(output, target).item()
            # 计算准确率
            pred = output.argmax(dim=1, keepdim=True) #获取output 最大值的索引，即标签。
            correct += pred.eq(target.view_as(pred)).sum().item()
#             .sum() 是计算张量所有元素的和，.item() 将结果转换为 Python 数值类型。pred.eq(target.view_as(pred)) 是计算预测值 pred 是否等于真实标签 target，返回一个张量，元素为 0 或 1。.sum() 将所有 1 累加起来，得到预测正确的数量。.item() 将结果转换为 Python 数值类型，以便后续计算准确率。
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


