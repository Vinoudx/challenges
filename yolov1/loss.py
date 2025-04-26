from torch import nn

from utils import *


class yololoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mse_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss()
        self.lambda_cord = 5
        self.lambda_noobj = 0.5

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        prediction = prediction.reshape([-1, self.S, self.S, self.C + self.B * 5])
        iou1 = iou(prediction[..., 21:25], target[..., 21:25])
        iou2 = iou(prediction[..., 26:30], target[..., 26:30])
        ious = torch.cat([iou1.unsqueeze(-1), iou2.unsqueeze(-1)], dim=-1)

        # 取iou最大的框作为这个cell中有效框
        mask = torch.argmax(ious, dim=-1, keepdim=False)
        # 区别背景框和物体框
        ground_truth = target[..., 20:21]

        pred_box = ground_truth * (prediction[..., 21:25] * (1 - mask) + prediction[..., 26:30] * mask)
        target_box = ground_truth * target[..., 21:25]
        pred_box[..., 2:4] = torch.sign(pred_box[..., 2:4]) * torch.sqrt(pred_box[..., 2:4] + 1e-6)
        target_box[..., 2:4] = torch.sqrt(target_box[..., 2:4])

        # cords loss ---
        cord_loss = self.mse_loss(torch.flatten(pred_box[..., 0:2], end_dim=-2),
                                  torch.flatten(target_box[..., 0:2], end_dim=-2))
        print("cord loss: {}".format(cord_loss))
        # w,h loss
        wh_loss = self.mse_loss(torch.flatten(pred_box[..., 2:4], end_dim=-2),
                                torch.flatten(target_box[..., 2:4], end_dim=-2))
        print("wh loss: {}".format(wh_loss))

        # obj loss ---

        pred_obj = ground_truth * (prediction[..., 20:21] * (1 - mask) + prediction[..., 25:26] * mask)
        target_obj = ground_truth * target[..., 20:21]

        obj_loss = self.mse_loss(torch.flatten(pred_obj[..., 0:1], end_dim=-2),
                                 torch.flatten(target_obj[..., 0:1], end_dim=-2))
        # print(torch.flatten(target_obj[..., 0:1], end_dim=-2))
        print("obj loss: {}".format(obj_loss))

        # noobj loss ---

        pred_noobj1 = (1 - ground_truth) * prediction[..., 20:21]
        pred_noobj2 = (1 - ground_truth) * prediction[..., 25:26]
        target_noobj = (1 - ground_truth) * target[..., 20:21]

        noobj_loss = (self.mse_loss(torch.flatten(pred_noobj1, end_dim=-2), torch.flatten(target_noobj, end_dim=-2)) +
                      self.mse_loss(torch.flatten(pred_noobj2, end_dim=-2), torch.flatten(target_noobj, end_dim=-2)))
        # print(torch.flatten(pred_noobj1, end_dim=-2))
        print("noobj loss: {}".format(noobj_loss))

        # class loss ---
        pred_class = ground_truth * prediction[..., :20]
        target_class = ground_truth * prediction[..., :20]
        class_loss = self.class_loss(torch.flatten(pred_class, end_dim=-2),
                                     torch.flatten(target_class[..., :20], end_dim=-2))
        print("class loss: {}".format(class_loss))
        return cord_loss * self.lambda_cord + wh_loss + obj_loss + noobj_loss * self.lambda_noobj + class_loss
