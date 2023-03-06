import numpy as np

def xcycwh_to_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """Converts the x_center, y_center to x_1, y_1 (top left point) in a vectorized way.

    Args:
        bboxes (np.ndarray): The bounding boxes of xcyc format.

    Returns:
        np.ndarray: The bounding boxes of x1y1 format.
    """
    bboxes[:,:2] = bboxes[:,:2] - (bboxes[:, 2:] / 2)
    bboxes[:, 2:] = bboxes[:,:2] + bboxes[:, 2:]
    
    return bboxes


def x1y1_to_xcyc(bboxes: np.ndarray) -> np.ndarray:
    """Converts the x_1, y_1 (top left point) to x_center, y_center in a vectorized way.

    Args:
        bboxes (np.ndarray): The bounding boxes of x1y1 format.

    Returns:
        np.ndarray: The bounding boxes of xcyc format.
    """
    
    bboxes[:,:2] = bboxes[:,:2] + (bboxes[:, 2:] / 2)
    
    return bboxes


def x1y1wh_to_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """Converts the xywh to xyxy in a vectorized way.

    Args:
        bboxes (np.ndarray): The bounding boxes with xywh format.

    Returns:
        np.ndarray: The bounding boxes with xyxy format.
    """
    
    bboxes[:, 2:] = bboxes[:,:2] + bboxes[:, 2:]
    
    return bboxes


def xyxy_to_x1y1wh(bboxes: np.ndarray) -> np.ndarray:
    """Converts the xyxy to xywh in a vectorized way.

    Args:
        bboxes (np.ndarray): The bounding boxes with xyxy format.

    Returns:
        np.ndarray: The bounding boxes with xywh format.
    """
    
    bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:,:2]
    
    return bboxes


def normalize_bboxes(bboxes: np.ndarray, img_width: float, img_height: float) -> np.ndarray:
    """Normalizes the bounding boxes with respect to image width and height.

    Args:
        bboxes (np.ndarray): The bounding boxes to be normalizes.
        img_width (float): The width of the image.
        img_height (float): The height of the image.

    Returns:
        np.ndarray: The normalized bounding box coordinates.
    """
    
    bboxes[:, 0] /= img_width
    bboxes[:, 1] /= img_height
    bboxes[:, 2] /= img_width
    bboxes[:, 3] /= img_height
    
    return bboxes


def resize_bboxes(bboxes: np.ndarray, ratio: float) -> np.ndarray:
    """Resizes the bounding boxes in order to fit the image again.

    Args:
        bboxes (np.ndarray): Boinding boxes to be resized.
        ratio (float): The resize ratio.

    Returns:
        np.ndarray: The resized bounding boxes.
    """
    
    bboxes *= ratio
    
    return bboxes


def adjust_bboxes(bboxes: np.ndarray, padw: float, padh: float) -> np.ndarray:
    """Adjust bounding boxes in order to letter boxes image again.

    Args:
        bboxes (np.ndarray): The bounding boxes.
        padw (float): The padding value of the image width.
        padh (float): The padding value of the image heigth.

    Returns:
        np.ndarray: Adjusted bounding boxes.
    """
    
    bboxes[:, 0] += padw
    bboxes[:, 1] += padh
    bboxes[:, 2] += padw
    bboxes[:, 3] += padh
    
    return   bboxes