import numpy as np
import torch
import torchvision
from process_box import xcycwh_to_xyxy


def non_max_suppression(prediction, conf_thres=0.9, iou_thres=0.6, classes=None, agnostic=False, multi_label=True,
                        max_det=300):
    
    num_classes = prediction.shape[2] - 5  # number of classes
    
    pred_candidates = prediction[..., 4] > conf_thres  # mask objectness scores.
    
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    
    max_nms = 300  # maximum number of boxes into torchvision.ops.nms()
    max_wh = 4096  # A safe number to avoid overlapping bounding boxes with different classes.
    
    multi_label &= num_classes > 1  # multiple labels per box
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    
    for pred_idx, pred in enumerate(prediction):  #for each batch
        pred = pred[pred_candidates[pred_idx]]  # confidence mask
        
        if not pred.shape[0]:
            continue
        
        pred[:, 5:] *= pred[:, 4:5]  # confidence_score = cls_confidences * objectness_score

        box = xcycwh_to_xyxy(pred[:, :4])

        
        if multi_label:
            i, j = (pred[:, 5:] > conf_thres).nonzero(as_tuple=False).T # index values of class confidences that higher than conf_thres
            pred = torch.cat((box[i], pred[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = pred[:, 5:].max(1, keepdim=True)
            pred = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        
        if classes is not None:
            pred = pred[(pred[:, 5:6] == torch.tensor(classes, device=pred.device)).any(1)]
        
        num_boxes = pred.shape[0]
        if not num_boxes:
            continue
        elif num_boxes > max_nms:  # epred_candidatesess boxes
            pred = pred[pred[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidenum_classese

        # Batched NMS
        cls = pred[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = pred[:, :4] + cls, pred[:, 4]  # boxes (offset by class (to have non overlapping bboxes with different classes)), scores 

        i = torchvision.ops.nms(boxes, scores, iou_thres)  # int64 tensor with the indices of the elements that have been kept by NMS, sorted in decreasing order of scores
        
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        output[pred_idx] = pred[i]

    return output

