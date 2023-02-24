import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage
import numpy as np
import random
import cv2

#TODO handle no bbox case

class Augmentations:
    
    def __init__(self, opt) -> None:
        
        opt = opt['imgaug']
        
        self.debug = False
        num_aug = opt['num_aug']
        
        augmentations = []
        
        if opt['fliplr']:   
            
            augmentations.append(iaa.Fliplr(opt['fliplr']))
            
        if opt['scale']:
            
            augmentations.append(iaa.Affine(scale=opt['scale']))
            
        if opt['brightness']:   
            
            augmentations.append(iaa.AddToBrightness(opt['brightness']))
            
        if opt['saturation']:  
             
            augmentations.append(iaa.AddToSaturation(opt['saturation']))
            
        if opt['hue']:   
            
            augmentations.append(iaa.AddToHue(opt['hue']))
            
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
            
            self.seq = iaa.SomeOf(n=num_aug, children=augmentations)
        
    def __call__(self, img_data):
        
        img = img_data['img']
        bboxes = np.array(img_data['labels'][:,:4])
        category_ids = img_data['labels'][:,4]

        bboxes_iaa = BoundingBoxesOnImage([], img.shape).from_xyxy_array(bboxes, img.shape)
        img, bbox_aug = self.seq(image=img.astype(np.uint8), bounding_boxes=bboxes_iaa)

        bboxes = bbox_aug.to_xyxy_array()
        bboxes[bboxes<0] = 0

        if self.debug:
            
            bboxes_iaa = BoundingBoxesOnImage([], img.shape).from_xyxy_array(bboxes, img.shape)
            image_after = bboxes_iaa.draw_on_image(img, size=2)
            cv2.imwrite('image_imgaug.jpg', image_after)

        labels = np.concatenate((bboxes, np.expand_dims(category_ids, 1)),1)

        img_data = {'img': img, 'labels': labels}
        
        return img_data
    
    
class CopyPaste:
    
    def __init__(self, opt=None) -> None:
        
        self.debug = False
        opt = opt['copy_paste']
        self.ioa_thr = opt['box_augments']['threshold']
        
        self.bboxes_len = opt['bboxes_memory']
        self.paste_len = opt['pasted_bbox_number']
        self.boxes_w_labels = np.array([])

        if opt['augment_box']:
            
            bbox_augmentations = []
            
            if opt['box_augments']['fliplr']:  
                 
                bbox_augmentations.append(iaa.Fliplr(opt['box_augments']['fliplr']))
                
            if opt['box_augments']['brightness']:   
                
                bbox_augmentations.append(iaa.AddToBrightness(opt['box_augments']['brightness']))
                
            if opt['box_augments']['saturation']:   
                
                bbox_augmentations.append(iaa.AddToSaturation(opt['box_augments']['saturation']))
                
            if opt['box_augments']['hue']:  
                 
                bbox_augmentations.append(iaa.AddToHue(opt['box_augments']['hue']))
                
            if opt['box_augments']['add_grayscale']: 
                 
                bbox_augmentations.append(iaa.Grayscale(opt['box_augments']['add_grayscale']))
                
            if opt['box_augments']['motion_blur']:   
                
                bbox_augmentations.append(iaa.MotionBlur(opt['box_augments']['motion_blur']))
                
            if opt['box_augments']['contrast']:
                
                bbox_augmentations.append(iaa.GammaContrast(opt['box_augments']['contrast'], per_channel=True))

            self.bbox_augmentations = iaa.SomeOf(n=1, children=bbox_augmentations)

        else:
            
            self.bbox_augmentations = None


    def __call__(self, img_data):

        img = img_data['img']
        bboxes = img_data['labels'][:,:4]
        category_ids = img_data['labels'][:,4]
        
        cropped_boxes = self.get_bbox(img, bboxes, category_ids)
        
        if not len(self.boxes_w_labels):
            
            self.boxes_w_labels = cropped_boxes
            self.check_bbox_memory()
            
            return img_data
            
        if len(self.boxes_w_labels) < self.paste_len:
            
            bbox_number = len(self.boxes_w_labels)
            
        else:
            
            bbox_number = random.randint(1, self.paste_len)

        pasted_boxes = self.boxes_w_labels[:,1][:bbox_number]

        if self.bbox_augmentations:
            
            pasted_boxes = self.bbox_augmentations(images=pasted_boxes)
            pasted_boxes = np.array([np.squeeze(x) for x in pasted_boxes], dtype=object)

        pasted_cat_ids = self.boxes_w_labels[:,0][:bbox_number]
        self.boxes_w_labels = self.boxes_w_labels[bbox_number:]

        new_boxes = []
        new_category_ids = []
        area_1 = calc_bbox_area(bboxes=bboxes)
        area_2 = calc_img_area(images=pasted_boxes)
        ratio = min(area_1 / area_2, 1)
        areas = [x.shape[0] * x.shape[1] for x in pasted_boxes]
        grids = [[[0, img.shape[0] / 2], [0, img.shape[1] / 2]],
                 [[0, img.shape[0] / 2], [img.shape[1] / 2, img.shape[1]]],
                 [[img.shape[0] / 2, img.shape[0]], [0, img.shape[1] / 2]],
                 [[img.shape[0] / 2, img.shape[0]], [img.shape[1] / 2, img.shape[1]]]]
        grid_idx = 0

        for _, bbox in sorted(zip(areas, pasted_boxes), key=lambda x: x[0], reverse=True):
            
            grid = grids[grid_idx]
            gridx = grid[1]
            gridy = grid[0]
            
            if ratio < 0.1:
                
                continue
            
            try:
                
                bbox=cv2.resize(bbox, (0,0), fx=ratio, fy=ratio)
                
            except:
                
                pass
            
            if int(gridy[1] - bbox.shape[0] - 1) > int(gridy[0]):
                
                y1 = random.randrange(int(gridy[0]), int(gridy[1] - bbox.shape[0] - 1))
                
            elif img.shape[0] - bbox.shape[0] - 1 > 0:
                
                y1 = random.randrange(img.shape[0] - bbox.shape[0] - 1)
                
            else:
                
                continue

            
            if int(gridx[1] - bbox.shape[1] - 1) > int(gridx[0]):
                
                x1 = random.randrange(int(gridx[0]), int(gridx[1] - bbox.shape[1] - 1))
                
            elif (img.shape[1] - bbox.shape[1] - 1) > 0:
                
                x1 = random.randrange(img.shape[1] - bbox.shape[1] - 1)
                
            else:
                
                continue
            

            y2 = int(y1 + bbox.shape[0])
            x2 = int(x1 + bbox.shape[1])
            img[y1:y2, x1:x2, ...] = bbox
            new_box = np.array([x1, y1, x2, y2])
            new_boxes.append(new_box)
            new_category_ids.append(pasted_cat_ids[grid_idx])
            grid_idx += 1

        if not len(new_boxes):
            
            return img_data
        
        new_boxes = np.array(new_boxes)
        ioas = inter_over_area(bboxes, new_boxes)

        i = 0
        ioas = np.sum(ioas, axis=1)

        for iou in ioas:
            
            if np.any(iou>self.ioa_thr):
                
                bboxes = np.delete(bboxes, i, axis=0)
                category_ids = np.delete(category_ids, i, axis=0)
                
            else:
                
                i += 1
                
        bboxes = np.concatenate((bboxes, new_boxes), 0)
        category_ids = np.concatenate((category_ids, new_category_ids), 0)
        
        if self.debug:
            
            bboxes_iaa = BoundingBoxesOnImage([], img.shape).from_xyxy_array(bboxes, img.shape)
            image_after = bboxes_iaa.draw_on_image(img, size=2)
            cv2.imwrite('image_copypaste.jpg', image_after)

        self.boxes_w_labels = np.concatenate((self.boxes_w_labels, cropped_boxes), 0)
        self.check_bbox_memory()
        self.shuffle_bboxes()

        labels = np.concatenate((bboxes, np.expand_dims(category_ids, 1)),1)
        img_data = {'img': img, 'labels': labels}
        
        return img_data



    def check_bbox_memory(self):
        
        if len(self.boxes_w_labels) > self.bboxes_len:

            diff = len(self.boxes_w_labels) - self.bboxes_len

            self.boxes_w_labels = self.boxes_w_labels[diff:]
            
    def shuffle_bboxes(self):
        
        np.random.shuffle(self.boxes_w_labels)


    def get_bbox(self, img, bboxes, category_id):

        cropped_bbox = []

        for i, bbox in enumerate(bboxes):

            cropped_bbox.append([int(category_id[i]), self.crop_bbox(img, bbox, i)])
            

        return np.array(cropped_bbox, dtype=object)

    def crop_bbox(self, img, bbox, i):
        
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        cropped_bbox = img[y1:y2, x1:x2, ...].copy()

        return cropped_bbox



def calc_bbox_area(bboxes):
    
    return np.max((bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1]))

def calc_img_area(images):
    
    max_area = 0
    
    for img in images:

        area = img.shape[0] * img.shape[1]
        
        if area > max_area:
            
            max_area = area
        
    return max_area
    
    
def inter_over_area(bboxes1, bboxes2):

    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    ioa = interArea / boxAArea

    return ioa


class CutOut:

    def __init__(self, opt):
        self.debug = False
        opt = opt['cutout']
        percentages = opt['percentages']
        fill_type = opt['fill_type']
        self.ioa_thr = opt['threshold']
        self.percentages = sorted(percentages, reverse=True)
        types = ['gaussian_noise', 'random_color', 'white', 'black', 'gray']
        self.fill_type = types[fill_type]

    def __call__(self, img_data):

        img = img_data['img']
        bboxes = img_data['labels'][:,:4]
        category_ids = img_data['labels'][:,4]
        
        height, width = img.shape[:2]

        
        cutout_boxes = []
        
        for scale in self.percentages:
            
            h, w = int(height * scale), int(width * scale)
            y1 = int(random.randrange(height - h - 1))
            x1 = int(random.randrange(width - w - 1))
            y2 = y1 + h
            x2 = x1 + w
            
            if self.fill_type == 'random_color':
                
                box = np.full((h, w, 3), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                
            elif self.fill_type == 'white':
                
                box = np.full((h, w, 3), 255)
                
            elif self.fill_type == 'black':
                
                box = np.full((h, w, 3), 0)
                
            elif self.fill_type == 'gaussian_noise':
                
                box = np.random.normal(0, 1.5, (h, w, 3)) * 255
                
            elif self.fill_type == 'gray':
                
                box = np.full((h, w, 3), 127)

            img[y1:y2, x1:x2, :] = box
            cutout_boxes.append(np.array([x1, y1, x2, y2]))

        cutout_boxes = np.array(cutout_boxes)
        ioas = inter_over_area(bboxes, cutout_boxes)
        ioas = np.sum(ioas, axis=1)

        i = 0
        
        for ioa in ioas:
            
            if np.any(ioa>self.ioa_thr):
                
                bboxes = np.delete(bboxes, i, axis=0)
                category_ids = np.delete(category_ids, i, axis=0)
                
            else:
                
                i += 1

        if self.debug:
            
            bboxes_iaa = BoundingBoxesOnImage([], img.shape).from_xyxy_array(bboxes, img.shape)
            image_after = bboxes_iaa.draw_on_image(img, size=2)
            cv2.imwrite('image_cutout.jpg', image_after)
            
        labels = np.concatenate((bboxes, np.expand_dims(category_ids, 1)),1)
        img_data = {'img': img, 'labels': labels}
        
        return img_data


def get_augmentations(opt):
    augmentations = []
    aug_names = opt['training']['augmentations']

    for name in aug_names:
        if name == 'imgaug':
            augmentations.append(Augmentations(opt))
        elif name == 'copy_paste':
            augmentations.append(CopyPaste(opt))
        elif name == 'cutout':
            augmentations.append(CutOut(opt))
    
    return augmentations
