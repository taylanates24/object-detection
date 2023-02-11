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
    
    