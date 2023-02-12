from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
from augmentations import HorizontalFlip

class CustomDataset(Dataset):

    def __init__(self, image_path, annotation_path, image_size=640, normalize=False, augment=False, augmentations=None) -> None:
        super(CustomDataset, self).__init__()
        
        self.image_root = os.path.join('/', *image_path.split('/')[:-1])
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.normalize = normalize
        self.coco = COCO(self.annotation_path)
        self.image_paths = sorted(os.listdir(self.image_path))
        self.ids = sorted(list(self.coco.imgs.keys()))
        self.image_size = image_size
        self.augmentations = augmentations
        
        if self.normalize:
            mean, stddev = self.get_statistics()
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=stddev)
            ])
        
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
            
    
    def __len__(self):
        
        return len(self.image_paths)
    
    
    def __getitem__(self, index):
        
        img, ratio, padding_w, padding_h = self.load_image(index)
        image_id = self.ids[index]
        
        bboxes, category_ids = self.load_labels(image_id=image_id,
                                  padh=padding_h, 
                                  padw=padding_w, 
                                  ratio=ratio)

        if self.augmentations:
            
            bboxes = self.x1y1wh_to_xyxy(bboxes=bboxes)
            img_data = {'img': img, 'bboxes':bboxes}
            
            for augment in self.augmentations:
                img_data = augment(img_data)
            
            img, bboxes = img_data['img'], img_data['bboxes']

            bboxes = self.xyxy_to_x1y1wh(bboxes=bboxes)

        bboxes = self.x1y1_to_xcyc(bboxes=bboxes)
        bboxes = self.normalize_bboxes(bboxes=bboxes, img_width=img.shape[1], img_height=img.shape[0])
        
        labels = np.concatenate((np.expand_dims(category_ids, 1), bboxes),1)
        
        if not self.normalize:
            img = img / 255
            
            

        img = self.transform(img) 
 
        return img, labels
    
    
    def load_image(self, index):
        
        img_path = self.image_paths[index]
        img = cv2.imread(os.path.join(self.image_path, img_path))
        height, width = img.shape[:2]
        ratio = self.image_size / max(height, width)            
        
        if ratio != 1:
            
            img = cv2.resize(img, (int(width*ratio), int(height*ratio)), interpolation=cv2.INTER_CUBIC)
        
        if img.shape[0] != img.shape[1]:
            
            img, padding_w, padding_h = self.letter_box(img=img, size=self.image_size)


        return img, ratio, padding_w, padding_h
        
    def load_labels(self, image_id, padh, padw, ratio):
        
        
        
        annotations = self.coco.imgToAnns[image_id]
        
        bboxes = []
        category_ids = []
        
        for ann in annotations:
            
            bboxes.append(ann['bbox'])
            category_ids.append(ann['category_id'])

        bboxes = np.array(bboxes)
        
        bboxes = self.adjust_bboxes(bboxes=bboxes,
                                    ratio=ratio,
                                    padw=padw,
                                    padh=padh)
        
        
        category_ids = np.array(category_ids)

            
        return bboxes, category_ids
        
        
    def adjust_bboxes(self, bboxes, ratio, padw, padh):
        
        bboxes[:, 0] = (bboxes[:, 0] * ratio + padw)
        bboxes[:, 1] = (bboxes[:, 1] * ratio + padh)
        bboxes[:, 2] = (bboxes[:, 2] * ratio)
        bboxes[:, 3] = (bboxes[:, 3] * ratio)
        
        return   bboxes
    
    
    def letter_box(self, img, size):
        
        box = np.full([size, size, img.shape[2]], 127)
        h, w = img.shape[:2]
        h_diff = size - h
        w_diff = size - w
        
        if h_diff:
            
            box[int(h_diff/2):int(img.shape[0]+h_diff/2), :img.shape[1], :] = img
 
        else:
            
            box[:img.shape[0], int(w_diff/2):int(img.shape[1]+w_diff/2), :] = img
        
        return box, w_diff / 2, h_diff / 2
        
        
        
    def get_statistics(self):
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_data = torchvision.datasets.ImageFolder(root=self.image_root, transform=transform)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)

        mean = 0.
        std = 0.
        nb_samples = 0.
        
        for data, _ in tqdm(train_data_loader):
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples

        return mean, std

    def x1y1_to_xcyc(self, bboxes):
        
        bboxes[:,:2] = bboxes[:,:2] + (bboxes[:, 2:] / 2)
        
        return bboxes
 
    def x1y1wh_to_xyxy(self, bboxes):
        
        bboxes[:, 2:] = bboxes[:,:2] + bboxes[:, 2:]
        
        return bboxes
    
    def xyxy_to_x1y1wh(self, bboxes):
        
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:,:2]
        
        return bboxes
    
    def normalize_bboxes(self, bboxes, img_width, img_height):
        
        bboxes[:, 0] /= img_width
        bboxes[:, 1] /= img_height
        bboxes[:, 2] /= img_width
        bboxes[:, 3] /= img_height
        return bboxes
