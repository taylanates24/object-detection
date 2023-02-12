import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage
import numpy as np

class Augmentations:
    
    def __init__(self, prob=0.5, scale=[0.95, 1.05], brightness=[-10, 10], saturation=[-10, 10], hue=[-10, 10], add_grayscale=[0, 0.2],
                 motion_blur=[3,5], contrast=[0.8,1.2], translate=[[-0.1,0.1],[-0.1,0.1]], rotate=[-5,5], shear=[-5,5]) -> None:
        
        fliplr = iaa.Fliplr(prob)
        scale = iaa.Affine(scale=scale)
        brightness = iaa.AddToBrightness(brightness)
        saturation = iaa.AddToSaturation(saturation)
        hue = iaa.AddToHue(hue)
        grayscale = iaa.Grayscale(alpha=add_grayscale)
        motion_blur = iaa.MotionBlur(k=motion_blur)
        contrast_shift = iaa.GammaContrast(contrast)
        translate = iaa.Affine(translate_percent={"x": translate[0], "y": translate[1]})
        rotate = iaa.Affine(rotate=rotate)
        shear = iaa.Affine(shear=shear)
        
        self.seq = iaa.Sequential([
            fliplr,
            scale,
            brightness,
            saturation,
            hue,
            grayscale,
            motion_blur,
            contrast_shift,
            translate,
            rotate,
            shear
            ])
        
    def __call__(self, img_data):
        
        img = img_data['img']
        bboxes = img_data['bboxes']
        
        bboxes_iaa = BoundingBoxesOnImage([], img.shape).from_xyxy_array(bboxes, img.shape)

        img, bbox_aug = self.seq(image=img.astype(np.uint8), bounding_boxes=bboxes_iaa)

        bboxes = bbox_aug.to_xyxy_array()
        bboxes[bboxes<0] = 0
        img_data = {'img': img, 'bboxes': bboxes}
        
        return img_data
        
        
class RandomAffine:
    
    def __init__(self) -> None:
        
        self.aug_scale = iaa.Affine(scale=(0.5, 1.5))
    
    
    def __call__(self, img_data):
        
        pass

    
    
class CopyPast:
    
    def __init__(self) -> None:
        pass
    
    
    def __call__(self, img, bboxes):
        #https://github.com/conradry/copy-paste-aug/blob/main/copy_paste.py
        pass