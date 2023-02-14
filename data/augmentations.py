import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage
import numpy as np
import random
class Augmentations:
    
    def __init__(self, opt) -> None:

        augmentations = []
        
        if opt['fliplr'] and random.random() < opt['fliplr'][-1]:   
            augmentations.append(iaa.Fliplr(opt['fliplr'][0]))
        if opt['scale'] and random.random() < opt['scale'][-1]:
            augmentations.append(iaa.Affine(scale=opt['scale'][:2]))
        if opt['brightness'] and random.random() < opt['brightness'][-1]:   
            augmentations.append(iaa.AddToBrightness(opt['brightness'][:2]))
        if opt['saturation'] and random.random() < opt['saturation'][-1]:   
            augmentations.append(iaa.AddToSaturation(opt['saturation'][:2]))
        if opt['hue'] and random.random() < opt['hue'][-1]:   
            augmentations.append(iaa.AddToHue(opt['hue'][:2]))
        if opt['add_grayscale'] and random.random() < opt['add_grayscale'][-1]:   
            augmentations.append(iaa.Grayscale(alpha=opt['add_grayscale'][:2]))
        if opt['motion_blur'] and random.random() < opt['motion_blur'][-1]:   
            augmentations.append(iaa.MotionBlur(k=opt['motion_blur'][:2]))
        if opt['translate'] and random.random() < opt['translate'][-1]:   
            augmentations.append(iaa.Affine(translate_percent={"x": opt['translate'][0], "y": opt['translate'][1]}))
        if opt['rotate'] and random.random() < opt['rotate'][-1]:   
            augmentations.append(iaa.Affine(rotate=opt['rotate'][:2]))
        if opt['shear'] and random.random() < opt['shear'][-1]:   
            augmentations.append(iaa.Affine(shear=opt['shear'][:2]))

        if len(augmentations):
            self.seq = iaa.Sequential(augmentations)
        
    def __call__(self, img_data):
        
        img = img_data['img']
        bboxes = img_data['bboxes']
        
        bboxes_iaa = BoundingBoxesOnImage([], img.shape).from_xyxy_array(bboxes, img.shape)

        img, bbox_aug = self.seq(image=img.astype(np.uint8), bounding_boxes=bboxes_iaa)

        bboxes = bbox_aug.to_xyxy_array()
        bboxes[bboxes<0] = 0
        img_data = {'img': img, 'bboxes': bboxes}
        
        return img_data
    
    
class CopyPast:
    
    def __init__(self) -> None:
        pass
    
    
    def __call__(self, img, bboxes):
        #https://github.com/conradry/copy-paste-aug/blob/main/copy_paste.py
        pass