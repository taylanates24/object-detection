from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
from augmentations import Augmentations, CopyPaste, CutOut
from process_box import x1y1_to_xcyc, x1y1wh_to_xyxy, xyxy_to_x1y1wh, normalize_bboxes, resize_bboxes, adjust_bboxes



class CustomDataset(Dataset):

    def __init__(self, image_path, annotation_path, image_size=640, normalize=True, augment=False, augmentations=None) -> None:
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
            
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),   
                    transforms.Normalize(mean=mean, std=stddev)
                ]
            )
       
        else:
            
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
            
    
    def __len__(self):
        
        return len(self.ids)
    
    
    def __getitem__(self, index):
        
        
        image_id = self.ids[index]

        img, ratio = self.load_image(image_id)
        labels = self.load_labels(image_id=image_id, ratio=ratio)
        
        if self.augmentations:
            
            img_data = {'img': img, 'labels':labels}
            
            for augment in self.augmentations:
                
                img_data = augment(img_data)
            
            img, labels = img_data['img'], img_data['labels']

        if img.shape[0] != img.shape[1]:
            
            img, padding_w, padding_h = self.letter_box(img=img, size=self.image_size)
            labels[:, :4] = adjust_bboxes(bboxes=labels[:, :4],
                                        padw=padding_w,
                                        padh=padding_h)

        img = img.astype(np.float32) / 255.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            
            img = self.transform(img)

        sample = {'img': img, 'labels': labels, 'scale': ratio}
 
        return sample
    
    
    def load_image(self, img_id):
        
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.image_path, img_path))


        height, width = img.shape[:2]
        ratio = self.image_size / max(height, width)            
        
        if ratio != 1:
            
            img = cv2.resize(img, (int(width*ratio), int(height*ratio)), interpolation=cv2.INTER_CUBIC)

        return img, ratio
        
    def load_labels(self, image_id, ratio):
        
        
        
        annotations = self.coco.imgToAnns[image_id]
        
        bboxes = []
        category_ids = []
        
        for ann in annotations:
            
            bboxes.append(ann['bbox'])
            category_ids.append(ann['category_id'] - 1)

        bboxes = np.array(bboxes)
        
        if ratio != 1:
            
            resize_bboxes(bboxes=bboxes, ratio=ratio)
        
        category_ids = np.array(category_ids)

        labels = np.concatenate((bboxes, np.expand_dims(category_ids, 1)),1)
        labels[:, :4] = x1y1wh_to_xyxy(labels[:,:4])
        
        return labels
        
        

    
    
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

def collater(data):
    
    imgs = [s['img'] for s in data]
    annots = [torch.tensor(s['labels']) for s in data]
    scales = [s['scale'] for s in data]

    
    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    

    return {'img': imgs, 'labels': annot_padded, 'scale': scales}



