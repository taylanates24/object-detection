import numpy as np

def xcycwh_to_xyxy(bboxes: np.ndarray) -> np.ndarray:
    # Convert boxes from [xc, yc, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    bboxes[:,:2] = bboxes[:,:2] - (bboxes[:, 2:] / 2)
    bboxes[:, 2:] = bboxes[:,:2] + bboxes[:, 2:]
    
    return bboxes


def x1y1_to_xcyc(bboxes: np.ndarray) -> np.ndarray:
    
    bboxes[:,:2] = bboxes[:,:2] + (bboxes[:, 2:] / 2)
    
    return bboxes

def x1y1wh_to_xyxy(bboxes: np.ndarray) -> np.ndarray:
    
    bboxes[:, 2:] = bboxes[:,:2] + bboxes[:, 2:]
    
    return bboxes

def xyxy_to_x1y1wh(bboxes: np.ndarray) -> np.ndarray:
    
    bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:,:2]
    
    return bboxes

def normalize_bboxes(bboxes: np.ndarray, img_width: float, img_height: float) -> np.ndarray:
    
    bboxes[:, 0] /= img_width
    bboxes[:, 1] /= img_height
    bboxes[:, 2] /= img_width
    bboxes[:, 3] /= img_height
    
    return bboxes


def resize_bboxes(bboxes: np.ndarray, ratio: float) -> np.ndarray:
    
    bboxes *= ratio
    
    return bboxes

def adjust_bboxes(bboxes: np.ndarray, padw: float, padh: float) -> np.ndarray:
    
    bboxes[:, 0] += padw
    bboxes[:, 1] += padh
    bboxes[:, 2] += padw
    bboxes[:, 3] += padh
    
    return   bboxes