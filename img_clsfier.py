# an example from https://blog.csdn.net/mengxianglong123/article/details/126034288

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

transform = transforms.Compose(
    [
    transforms.ToTensor(),
    ## https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
    # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ## https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
    # Normalize a tensor image with mean and standard deviation. 
    # This transform does not support PIL Image. 
    # Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels, 
    # this transform will normalize each channel of the input torch.*Tensor 
    # i.e., output[channel] = (input[channel] - mean[channel]) / std[channel]
    ]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(trainloader)
# images, labels = dataiter.next()
### this doesnt work, the solution is the following line from https://stackoverflow.com/questions/74289077/attributeerror-multiprocessingdataloaderiter-object-has-no-attribute-next 
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print('x, size', x.size)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
# net = Net().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params = net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad() # has to zerofiy the gradients before each epoch occurs.
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 2000 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('finished training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)




dataiter = iter(testloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))







# 实例化模型的类对象
net = Net()
# 加载训练阶段保存好的模型的状态字典
net.load_state_dict(torch.load(PATH))
# 利用模型对图片进行预测
outputs = net(images)
# 共有10个类别, 采用模型计算出的概率最大的作为预测的类别
_, predicted = torch.max(outputs, 1)
# 打印预测标签的结果
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# net.to(device)
# # 将输入的图片张量和标签张量转移到GPU上
# inputs, labels = data[0].to(device), data[1].to(device)
