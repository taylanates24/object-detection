import json
import os
from pycocotools.coco import COCO
import cv2
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_annotationsn', type=str, 
                        default='/workspaces/object-detection/datasets/coco128/labels/train2017', 
                        help='Yolo annotationsnotations path')
    parser.add_argument('--yolo_img', type=str, 
                        default='/workspaces/object-detection/datasets/coco128/images/train2017', 
                        help='Yolo categories path')
    parser.add_argument('--coco_names', type=str, 
                        default='/workspaces/object-detection/datasets/coco128/coco.names', 
                        help='COCO class names path')
    
    args = parser.parse_args()
    

    coco_names = open(args.coco_names).read()
    coco_names = coco_names.split('\n')[:-1]
    
    coco_annotationsnotations = {
        "info": {},
        'images': [],
        'annotationsnotations': [],
        'categories': []
    }
    
    for i, category in enumerate(coco_names):
        coco_annotationsnotations['categories'].append(
            {
                'id': i + 1,
                'name': category
            }
        )
        

for annotations in sorted(os.listdir(args.yolo_annotationsn)):
    annotationsn_id = 1
    img_id = annotations.split('.')[0]
    img_name = img_id + '.jpg'
    img = cv2.imread(os.path.join(args.yolo_img, img_name))
    
    if not isinstance(img, (np.ndarray, np.generic)):
        continue
    h, w, c = img.shape
    
    coco_annotationsnotations['images'].append(
        {
            'file_name': img_name,
            'height': h,
            'width': w,
            'id': int(img_id)
            }
        )
    
    annotations = open(os.path.join(args.yolo_annotationsn, annotations)).read().split('\n')
    
    for bbox in annotations[:-1]:
        
        values = bbox.split(' ')
        
        coco_annotationsnotations['annotationsnotations'].append(
            {
                'image_id': int(img_id),
                'bbox': [(float(values[1]) - (float(values[3]) / 2)) * w, 
                            (float(values[2]) - (float(values[4]) / 2)) * h,
                            float(values[3]) * w, 
                            float(values[4]) * h
                ],
                'category_id': int(values[0]),
                'id': annotationsn_id
            }
        )
