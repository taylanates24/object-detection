from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
import os
import cv2
class CustomDataset(Dataset):

    def __init__(self, image_path, annotation_path, image_size=640, normalize=False, augment=False) -> None:
        super(CustomDataset, self).__init__()
        
        self.image_root = os.path.join('/', *image_path.split('/')[:-1])
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.normalize = normalize
        self.augment = augment
        self.coco = COCO(self.annotation_path)
        self.image_paths = sorted(os.listdir(self.image_path))
        self.ids = list(self.coco.imgs.keys())
        self.image_size = image_size
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
        img = self.load_image(index)

        img_tensor = self.transform(img)

        return img_tensor
    
    
    def load_image(self, index):
        img_path = self.image_paths[index]
        img = cv2.imread(os.path.join(self.image_path, img_path))
        height, widht = img.shape[:2]
        ratio = self.image_size / max(height, widht)
        
        if ratio != 1:
            img = cv2.resize(img, (int(widht*ratio), int(height*ratio)), interpolation=cv2.INTER_CUBIC)
        
        if img.shape[0] != img.shape[1]:
            img = self.letter_box(img=img, size=self.image_size)
        return img
        
        
    def letter_box(self, img, size):
        
        
        
        return img
        
        
        
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



