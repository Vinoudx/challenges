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


def nms(bboxes, iou_threshold, threshold):
    # bboxes (batch_size, S, S, 6)
    #       (class_type, confidence, x, y, w, h)

    bboxes = bboxes.reshape([-1, 6])
    # print(bboxes.shape)
    results = []
    boxes = [i for i in bboxes if i[1] > threshold]
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)

    while boxes:
        box = boxes.pop(0)
        boxes = [b for b in boxes if b[0] != box[0] or iou(b[2:], box[2:]) < iou_threshold]
        results.append(box)

    return results


def iou(bbox1, bbox2):
    # bbox (-1, 4) x, y, w, h

    tlx1, tly1 = bbox1[..., 0:1] - bbox1[..., 2:3] / 2, bbox1[..., 1:2] - bbox1[..., 3:4] / 2
    rbx1, rby1 = bbox1[..., 0:1] + bbox1[..., 2:3] / 2, bbox1[..., 1:2] + bbox1[..., 3:4] / 2
    tlx2, tly2 = bbox2[..., 0:1] - bbox2[..., 2:3] / 2, bbox2[..., 1:2] - bbox2[..., 3:4] / 2
    rbx2, rby2 = bbox2[..., 0:1] + bbox2[..., 2:3] / 2, bbox2[..., 1:2] + bbox2[..., 3:4] / 2

    tlx, tly = torch.max(tlx1, tlx2), torch.max(tly1, tly2)
    rbx, rby = torch.min(rbx1, rbx2), torch.min(rby1, rby2)

    w, h = (rbx - tlx).clamp(0), (rby - tly).clamp(0)
    intersection = w * h
    union = abs((rbx1 - tlx1) * (rby1 - tly1)) + abs((rbx2 - tlx2) * (rby2 - tly2)) - intersection + 1e-6

    return intersection / union


if __name__ == "__main__":
    a = torch.zeros([1, 6], dtype=torch.float32)
    b = torch.zeros([1, 6], dtype=torch.float32)
    a[..., :] = torch.tensor([1, 0.9, 0.5, 0.5, 0.4, 0.5])
    b[..., :] = torch.tensor([1, 0.9, 0.6, 0.55, 0.3, 0.4])
    print(a, b)
    c = iou(a[..., 2:6], b[..., 2:6])
    print(c)

    d = torch.cat([a, b], dim=0)
    print(d)
    e = nms(d, 0.4, 0.4)
    print(e)
