import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage
import numpy as np

class Augmentations:
    
    def __init__(self, opt) -> None:

        augmentations = []
        
        if opt['fliplr']:   
            augmentations.append(iaa.Fliplr(opt['fliplr']))
        if opt['scale']:   
            augmentations.append(iaa.Affine(scale=opt['scale']))
        if opt['brightness']:   
            augmentations.append(iaa.AddToBrightness(opt['brightness']))
        if opt['saturation']:   
            augmentations.append(iaa.AddToSaturation(opt['saturation']))
        if opt['add_grayscale']:   
            augmentations.append(iaa.Grayscale(alpha=opt['add_grayscale']))
        if opt['motion_blur']:   
            augmentations.append(iaa.MotionBlur(k=opt['motion_blur']))
        if opt['translate']:   
            augmentations.append(iaa.Affine(translate_percent={"x": opt['translate'][0], "y": opt['translate'][1]}))
        if opt['rotate']:   
            augmentations.append(iaa.Affine(rotate=opt['rotate']))
        if opt['shear']:   
            augmentations.append(iaa.Affine(shear=opt['shear']))

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