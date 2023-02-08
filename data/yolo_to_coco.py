import json
import os
from pycocotools.coco import COCO
import cv2
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_ann', type=str, 
                        default='/workspaces/object-detection/datasets/coco128/labels/train2017', 
                        help='Yolo annotations path')
    parser.add_argument('--yolo_img', type=str, 
                        default='/workspaces/object-detection/datasets/coco128/images/train2017', 
                        help='Yolo categories path')
    parser.add_argument('--coco_names', type=str, 
                        default='/workspaces/object-detection/datasets/coco128/coco.names', 
                        help='COCO class names path')
    
    args = parser.parse_args()
    

    coco_names = open(args.coco_names).read()
    coco_names = coco_names.split('\n')[:-1]
    
    coco_annotations = {
        "info": {},
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    for i, category in enumerate(coco_names):
        coco_annotations['categories'].append(
            {
                'id': i + 1,
                'name': category
            }
        )
        

for an in sorted(os.listdir(args.yolo_ann)):
    ann_id = 1
    img_id = an.split('.')[0]
    img_name = img_id + '.jpg'
    img = cv2.imread(os.path.join(args.yolo_img, img_name))
    
    if not isinstance(img, (np.ndarray, np.generic)):
        continue
    h, w, c = img.shape
    
    coco_annotations['images'].append(
        {
            'file_name': img_name,
            'height': h,
            'width': w,
            'id': int(img_id)
            }
        )
        