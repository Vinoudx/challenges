import torch


def pred_to_bbox(pred: torch.Tensor, S=7, B=2, C=20):
    if torch.cuda.is_available():
        pred.to("cpu")

    batch_size = pred.shape[0]
    pred = pred.reshape([batch_size, S, S, B * 5 + C])
    bbox1 = pred[..., 21:25]
    bbox2 = pred[..., 26:30]

    box_confidence = torch.cat([pred[..., 20:21], pred[..., 25:26]], dim=-1)

    best_box = box_confidence.argmax(dim=-1, keepdim=True)
    # best_boxes 1*7*7*4
    best_boxes = bbox1 * (1 - best_box) + bbox2 * best_box

    # for conversions below:
    # https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/utils.py

    cell_index = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_index)
    y = 1 / S * (best_boxes[..., 1:2] + cell_index.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    # print(x, x.shape)
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    # print(converted_bboxes, converted_bboxes.shape)
    predicted_class = pred[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(pred[..., 20], pred[..., 25]).unsqueeze(-1)
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )
    return converted_preds

def nms():
    pass

def iou():
    pass