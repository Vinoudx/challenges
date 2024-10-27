import os.path
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
import cv2
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class TrainData(Dataset):
    def __init__(self):
        self.images = []
        self.labels = []
        self.index = []
        train_data = pd.read_csv("./classify-leaves/train.csv")
        for path, label in train_data.values:
            image = cv2.imread(os.path.join("./classify-leaves", path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
            image = image.float()
            self.images.append(image)
            self.labels.append(label)

        self.labels = pd.get_dummies(self.labels)
        self.index = self.labels.columns.tolist()
        self.labels = torch.tensor(self.labels.values).float()
        print("load complete...")

    def __getitem__(self, item):
        return self.images[item], self.labels[item]

    def getIndex(self):
        return self.index

    def __len__(self):
        return len(self.images)


class TestData(Dataset):
    def __init__(self):
        self.images = []
        self.index = []
        self.path = []
        train_data = pd.read_csv("./classify-leaves/test.csv")
        for path in train_data.values:
            path = path[0]
            image = cv2.imread(os.path.join("./classify-leaves", path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
            image = image.float()
            self.images.append(image)
            self.path.append(path)

        print("load complete...")

    def __getitem__(self, item):
        return self.images[item], self.path[item]

    def getIndex(self):
        return self.index

    def __len__(self):
        return len(self.images)

class Bottleneck(nn.Module):
    expansion = 4  # 输出通道数扩展因子

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 定义 ResNet50 模型
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


epoch = 100
lr = 0.01
wd = 0.0001
data = TrainData()
indexx = data.getIndex()

def train():
    data = TrainData()
    train_loader = DataLoader(data, batch_size=16, shuffle=True)
    # data_pre = Test_Data()
    # pred_loader = DataLoader(data_pre, batch_size=1, shuffle=False)

    model = ResNet(Bottleneck, [3, 4, 6, 3], 176).cuda()
    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(epoch):
        eopch_loss = []
        for index, item in enumerate(train_loader):
            input, label = item
            input = input.cuda()
            label = label.cuda()
            optim.zero_grad()
            output = model(input)
            l = loss_func(output, label)
            l.backward()
            optim.step()
            eopch_loss.append(l)
        print(f"epoch{i}, loss{sum(eopch_loss) / len(eopch_loss)}")
        torch.save(model, f"./classify-leaves/{i}.pth")





def pred():
    model = torch.load("./classify-leaves/35.pth")
    model.eval()
    data_pre = TestData()
    pred_loader = DataLoader(data_pre, batch_size=1, shuffle=False)

    for index, item in enumerate(pred_loader):
        input, index = item
        input = input.cuda()
        output = model(input)
        print(index)
        c = indexx[torch.argmax(output)]

        df = {
            'image': index,
            'label': c
        }
        df = pd.DataFrame(df)
        df.to_csv("./classify-leaves/submission.csv", mode='a', index=False, header=False)

# model = ResNet(Bottleneck, [3, 4, 6, 3], 176).cuda()

pred()