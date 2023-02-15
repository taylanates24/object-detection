import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage
import numpy as np
import random
import cv2
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
    
    
class CopyPaste:
    
    def __init__(self, bboxes_len=20, paste_len=3) -> None:
        

        self.bboxes_len = bboxes_len
        self.paste_len = paste_len
        self.boxes_w_labels = np.array([])
    
    def __call__(self, img_data):




        img = img_data['img']
        bboxes = img_data['labels'][:,1:]
        category_ids = img_data['labels'][:,0]



        bboxes_iaa = BoundingBoxesOnImage([], img.shape).from_xyxy_array(bboxes, img.shape)
        image_before = bboxes_iaa.draw_on_image(img, size=2)
        cv2.imwrite('paste_before.jpg', image_before)
        
        cropped_boxes = self.get_bbox(img, bboxes, category_ids)
        
        print('cropped_boxes')
        if not len(self.boxes_w_labels):
            self.boxes_w_labels = cropped_boxes
            self.check_bbox_memory()
            return img_data
            
        
        
        

        
        if len(self.boxes_w_labels) < self.paste_len:
            bbox_number = len(self.boxes_w_labels)
        else:
            #bbox_number = random.randint(1, self.paste_len)
            bbox_number=4
            
        pasted_boxes = self.boxes_w_labels[:,1][:bbox_number]
        pasted_cat_ids = self.boxes_w_labels[:,0][:bbox_number]
        self.boxes_w_labels = self.boxes_w_labels[bbox_number:]
        print('pasted_boxes')
        # check
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
        print('grid_idx')
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
            #print('asd')
            grid_idx += 1
        print('pasted')
        if not len(new_boxes):
            return img_data
        new_boxes = np.array(new_boxes)
        ious = inter_over_union(bboxes, new_boxes)
        print('ious')
        i = 0
        ious = np.sum(ious, axis=1)
        print('np.sum')
        print(ious)
        print('len ious: ', len(ious))
        print('shape:', category_ids.shape)
        for iou in ious:
            print('for iou')
            if np.any(iou>.3):
                bboxes = np.delete(bboxes, i, axis=0)
                category_ids = np.delete(category_ids, i, axis=0)
            else:
                i += 1
        print('deleted')
        bboxes = np.concatenate((bboxes, new_boxes), 0)
        category_ids = np.concatenate((category_ids, new_category_ids), 0)
        bboxes_iaa = BoundingBoxesOnImage([], img.shape).from_xyxy_array(bboxes, img.shape)
        image_before = bboxes_iaa.draw_on_image(img, size=2)
        cv2.imwrite('paste.jpg', image_before)
        
        self.boxes_w_labels = np.concatenate((self.boxes_w_labels, cropped_boxes), 0)
        self.check_bbox_memory()
        self.shuffle_bboxes()
        
        print('shuffle_bboxes')
        labels = np.concatenate((np.expand_dims(category_ids, 1), bboxes),1)
        img_data = {'img': img, 'labels': labels}
        print('bboxes:', len(self.boxes_w_labels))
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
        cv2.imwrite('bboxes/' + str(i) + '_bbox.jpg', cropped_bbox)
        #print('asd')

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
    
    
def inter_over_union(bboxes1, bboxes2):

    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou   
