# an example from https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E5%9B%9B%E7%AB%A0/%E5%9F%BA%E7%A1%80%E5%AE%9E%E6%88%98%E2%80%94%E2%80%94FashionMNIST%E6%97%B6%E8%A3%85%E5%88%86%E7%B1%BB.html

import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from torchvision import datasets

import pandas as pd

import matplotlib.pyplot as plt

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

batch_size = 256
num_workers = 4
lr = 1e-4
epochs = 20

image_size = 28
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_size),
    transforms.ToTensor()
])

# first way of fetching data
train_data = datasets.FashionMNIST(root='./fashiondata', train=True, 
                                    download=True,
                                    transform=data_transform)

test_data = datasets.FashionMNIST(root='./fashiondata', train=False, 
                                    download=True,
                                    transform=data_transform)

# second way of data fetching
# deal with csv files by myself
# soure: https://www.kaggle.com/zalando-research/fashionmnist

class FMDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.images = df.iloc[:, 1:].values.astype(np.uint8)
        self.labels = df.iloc[:, 0].values

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28,28,1)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image/255., dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

train_df = pd.read_csv("./fashionm_csv/fashion-mnist_train.csv")
test_df = pd.read_csv("./fashionm_csv/fashion-mnist_test.csv")
train_data = FMDataset(train_df, data_transform)
test_data = FMDataset(test_df, data_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)


image, label = next(iter(train_loader))
print(image.shape, label.shape)
plt.imshow(image[0][0], cmap="gray")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32,64,5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*4*4)
        x = self.fc(x)

model = Net()
# nodel = model.cuda()
## lift the comment if cuda is available

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        # data, label = data.cuda(), label.cuda()
        ## lift the comment if cuda is available
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

def val(epoch):
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in train_loader:
            # data, label = data.cuda(), label.cuda()
            ## lift the comment if cuda is available
            output = model(data)
            preds = torch.argmax(output,1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)

    val_loss = val_loss/len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    print('Epoch: {} \tValidation Loss: {:.6f}'.format(epoch, val_loss, acc))
    
for epoch in range(1, epochs+1):
    train(epoch)
    val(epoch)

save_path = "./FashionModel.pkl"
torch.save(model, save_path)
