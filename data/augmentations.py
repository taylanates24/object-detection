import cv2
import imgaug.augmenters as aug

class HorizontalFlip:
    
    def __init__(self) -> None:
        pass
    
    
    def __call__(self, img, bboxes):
        
        pass
    
class BrightnessShift:
    
    def __init__(self) -> None:
        pass
    
    
    def __call__(self, img, bboxes):
        #aug = iaa.AddToBrightness((-30, 30))
        pass
    
    
    
class SaturationShift:
    
    def __init__(self) -> None:
        pass
    
    
    def __call__(self, img, bboxes):
        #aug = iaa.AddToSaturation((-50, 50))
        pass
    
class HueShift:
    
    def __init__(self) -> None:
        pass
    
    
    def __call__(self, img, bboxes):
        #aug = iaa.AddToHue((-50, 50))
        pass
    
    
class AddGrayscale:
    
    def __init__(self) -> None:
        pass
    
    
    def __call__(self, img, bboxes):
        #aug = iaa.Grayscale(alpha=(0.0, 1.0))
        pass
    
    
class ChangeColorTemperature:
    
    def __init__(self) -> None:
        pass
    
    
    def __call__(self, img, bboxes):
        #aug = iaa.ChangeColorTemperature((1100, 10000))
        pass
    


class MotionBlur:
    
    def __init__(self) -> None:
        pass
    
    
    def __call__(self, img, bboxes):
        #aug = iaa.MotionBlur(k=15)
        pass

'''     bbox_aug = []
        for bbox in bboxes:
            bbox_aug.append(BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[0]+bbox[2], y2=bbox[1]+bbox[3]))
        bbs = BoundingBoxesOnImage(bbox_aug, shape=(img_heigth, img_width, 3))'''