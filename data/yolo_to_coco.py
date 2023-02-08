import json
import os
from pycocotools.coco import COCO
import cv2
import numpy as np
import argparse
import os
import logging
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_annotations', type=str, 
                        default='/workspaces/object-detection/datasets/coco128/labels/train2017', 
                        help='Yolo annotationsnotations path')
    parser.add_argument('--yolo_img', type=str, 
                        default='/workspaces/object-detection/datasets/coco128/images/train2017', 
                        help='Yolo categories path')
    parser.add_argument('--coco_names', type=str, 
                        default='/workspaces/object-detection/datasets/coco128/coco.names', 
                        help='COCO class names path')
    parser.add_argument('--out_file_name', type=str, 
                        default='coco128_train.json', 
                        help='new annotations file name') 
    parser.add_argument('--check', type=str, 
                        default=True, 
                        help='Check if the new format fits COCO')    
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
        

    for annotations in tqdm(sorted(os.listdir(args.yolo_annotations))):
        annotationsn_id = 1
        img_id = annotations.split('.')[0]
        img_name = img_id + '.jpg'
        img = cv2.imread(os.path.join(args.yolo_img, img_name))
        
        if not isinstance(img, (np.ndarray, np.generic)):
            logging.warning('An annotation file was not found!')
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
        
        annotations = open(os.path.join(args.yolo_annotations, annotations)).read().split('\n')
        
        for bbox in annotations[:-1]:
            
            values = bbox.split(' ')
            
            coco_annotations['annotations'].append(
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
    out_path = os.path.join(args.yolo_annotations, 'coco_annotations')
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
        
    out_path = os.path.join(out_path, args.out_file_name)
    new_coco_ann_file = json.dumps(coco_annotations, indent=4)

    with open(out_path, "w") as outfile:
        outfile.write(new_coco_ann_file)

    if args.check:
        try:
            COCO(out_path)
        except:
            logging.error('An error occured!')
        logging.critical(' The annotations successfully converted to COCO format.')
