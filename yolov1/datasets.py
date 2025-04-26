import torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import os
import pandas as pd

from utils import *


class image_datas(Dataset):
    def __init__(self, root_path, type="train"):
        self.path = root_path
        self.S = 7
        self.B = 2
        self.C = 20
        self.images = []
        self.labels = []
        data_root_path = os.path.join(root_path, 'bananas_{}'.format(type))
        image_root_path = os.path.join(data_root_path, 'images')
        label_path = os.path.join(data_root_path, "label.csv")
        image_names = os.listdir(image_root_path)
        labels = pd.read_csv(label_path)
        for image_name in image_names:
            image_path = os.path.join(image_root_path, image_name)
            image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            self.images.append(image)
            # image (3*256*256)
            # labels: (num_objs, class, x, y, w, h)
            # label: (class, xmin, ymin, xmax, ymax)
            label = torch.tensor(labels[labels['img_name'] == image_name].iloc[..., 1:].values, dtype=torch.float32)
            # 转换成x,y,w,h
            image_width, image_height, _ = image.shape
            x = (label[..., 3] + label[..., 1]) / 2 / image_width
            y = (label[..., 4] + label[..., 2]) / 2 / image_height
            w = (label[..., 3] - label[..., 1]) / image_width
            h = (label[..., 4] - label[..., 2]) / image_height

            image = cv.resize(image, (224, 224))
            self.labels.append(torch.cat([label[..., 0], x, y, w, h]))
            # self.labels (s*s*(c+5*b))
            # print(torch.cat([label[..., 0], x, y, w, h]))

    def __getitem__(self, item):
        class_type, x, y, w, h = self.labels[item].tolist()
        i, j, = int(self.S * y), int(self.S * x)
        x_cell, y_cell = self.S * x - j, self.S * y - i
        width_cell, height_cell = (w * self.S, h * self.S)
        label = torch.zeros([self.S, self.S, self.C + 5 * self.B])
        bias = 0 if label[i, j, 20] == 0 else 5
        label[i, j, 20 + bias] = 1
        label[i, j, 21 + bias:25 + bias] = torch.tensor([x_cell, y_cell, width_cell, height_cell])
        label[i, j, int(class_type)] = 1
        print(x_cell, y_cell, width_cell, height_cell)
        return self.images[item], label

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    a = image_datas('../banana-detection/')
    image, label = a.__getitem__(0)

    box_index = label[..., 20] != 0
    box = label[box_index]
    t = torch.nn.Flatten(start_dim=0, end_dim=-1)(label).unsqueeze(0)
    b = pred_to_bbox(t)
    #print(b.shape)
    print(b)

    b = b.reshape([-1, 6])
    for i in b:
        if i[1] > 0.5:
            print(i)
            x, y, w, h = i[..., 2:6]

            image_width, image_height, _ = image.shape
            xx = x * image_width
            yy = y * image_height
            ww = w * image_width
            hh = h * image_height

            tlx = xx - ww / 2
            tly = yy - hh / 2
            brx = xx + ww / 2
            bry = yy + hh / 2
            print(image_width, image_height, x, tlx, y, tly, w, brx, h, bry)
            cv.rectangle(image, (int(tlx), int(tly)), (int(brx), int(bry)), (0, 0, 255))
            cv.imshow("1", image)
            cv.waitKey(10000)



