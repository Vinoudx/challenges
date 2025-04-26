import torch
import cv2 as cv
from torchvision import transforms

from model import *
from utils import *

model = yolov1()

check_point = torch.load("./checkpoint.pth")
model.load_state_dict(check_point['model_state_dict'])

path = "../banana-detection/bananas_val/images/1.png"

origin_image = cv.imread(path)
image_width, image_height, _ = origin_image.shape
image = cv.resize(origin_image, (224, 224))
image = transforms.Compose([transforms.ToTensor()])(image).unsqueeze(0)

model.eval()
output = model(image)

bboxes = pred_to_bbox(output)
print(bboxes)
bboxes = nms(bboxes, 0.9, 0)


for i in bboxes:

        x, y, w, h = i[..., 2:6]

        xx = x * image_width
        yy = y * image_height
        ww = w * image_width
        hh = h * image_height

        tlx = xx - ww / 2
        tly = yy - hh / 2
        brx = xx + ww / 2
        bry = yy + hh / 2
        # print(image_width, image_height, x, tlx, y, tly, w, brx, h, bry)
        cv.rectangle(origin_image, (int(tlx), int(tly)), (int(brx), int(bry)), (0, 0, 255))
cv.imshow("1", origin_image)
cv.waitKey(10000)



