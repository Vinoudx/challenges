import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset,dataloader

from utils import *
from loss import *
from datasets import *
from model import *

model = yolov1().cuda()

mydataset = image_datas('../banana-detection/')
mydataloader = DataLoader(mydataset, batch_size=32, shuffle=True)

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.05, momentum=0.9)
loss_fun = yololoss()
epoch = 5

# checkpoint = torch.load('checkpoint.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if torch.cuda.is_available():
    model.cuda()

for i in range(epoch):
    epoch_loss = []
    for index, item in enumerate(mydataloader):
        image, label = item
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        optimizer.zero_grad()
        output = model(image)
        loss = loss_fun(output, label)
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss)
    print("epoch:{}, loss{}\n".format(i, sum(epoch_loss) / len(epoch_loss)))

torch.save({
    'epoch': i,                      # 当前训练的轮数
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, './checkpoint.pth')

# checkpoint = torch.load('checkpoint.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']