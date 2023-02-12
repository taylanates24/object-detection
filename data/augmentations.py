import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage
import numpy as np

class Augmentations:
    
    def __init__(self, prob=0.5) -> None:
        self.fliplr = iaa.Fliplr(prob)

        self.seq = iaa.Sequential([
            self.fliplr,
        ])
        
    def __call__(self, img_data):
        
        img = img_data['img']
        bboxes = img_data['bboxes']
        
        bboxes_iaa = BoundingBoxesOnImage([], img.shape).from_xyxy_array(bboxes, img.shape)

        img, bbox_aug = self.seq(image=img.astype(np.uint8), bounding_boxes=bboxes_iaa)

        bboxes = bbox_aug.to_xyxy_array()
        img_data = {'img': img, 'bboxes': bboxes}
        
        return img_data
        
    
class RandomAffine:
    
    def __init__(self) -> None:
        
        self.aug_scale = iaa.Affine(scale=(0.5, 1.5))
    
    
    def __call__(self, img_data):
        
        
        
        
        #aug = iaa.Affine(scale=(0.5, 1.5))
        #aug = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        #aug = iaa.Affine(rotate=(-45, 45))
        #aug = iaa.Affine(shear=(-16, 16))
        #aug = iaa.TranslateX(percent=(-0.1, 0.1))
        #aug = iaa.TranslateY(percent=(-0.1, 0.1))
        #aug = iaa.Rotate((-45, 45))
        #aug = iaa.ShearX((-20, 20))
        #aug = iaa.ShearY((-20, 20))
        #
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
    
    
class ContrastShift:
    
    def __init__(self) -> None:
        pass
    
    
    def __call__(self, img, bboxes):
        #aug = iaa.GammaContrast((0.5, 2.0))
        pass

'''     bbox_aug = []
        for bbox in bboxes:
            bbox_aug.append(BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[0]+bbox[2], y2=bbox[1]+bbox[3]))
        bbs = BoundingBoxesOnImage(bbox_aug, shape=(img_heigth, img_width, 3))'''
        
        

    
    
class CopyPast:
    
    def __init__(self) -> None:
        pass
    
    
    def __call__(self, img, bboxes):
        #https://github.com/conradry/copy-paste-aug/blob/main/copy_paste.py
        pass