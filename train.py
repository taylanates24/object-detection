import pytorch_lightning as pl
import yaml
from my_model.model import TyNet
from utils.utils import init_weights
import argparse
from data.coco_dataset import CustomDataset, collater
from data.augmentations import get_augmentations
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from models.loss import FocalLoss
from models.utils import get_optimizer, get_scheduler
from models.detector import Detector
import torch 


if __name__ == '__main__':

    torch.cuda.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', type=str, default='training.yaml', help='training config file')
    parser.add_argument('--dataset_cfg', type=str, default='coco.yml', help='training config file')
    
    args = parser.parse_args()

    opt = args.train_cfg
    with open(opt, 'r') as config:
        opt = yaml.safe_load(config)

    dataset_opt = args.dataset_cfg
    with open(dataset_opt, 'r') as config:
        dataset_opt = yaml.safe_load(config)

    model = TyNet(compound_coef=0, num_classes=len(dataset_opt['obj_list']),
                                    ratios=eval(dataset_opt['anchors_ratios']), 
                                    scales=eval(dataset_opt['anchors_scales']))

    init_weights(model)
    model = model.cuda()

    augmentations = get_augmentations(opt)

    training_params = {'batch_size': opt['training']['batch_size'],
                        'shuffle': opt['training']['shuffle'],
                        'drop_last': opt['training']['drop_last'],
                        'collate_fn': collater,
                        'num_workers': opt['training']['num_workers']}

    val_params = {'batch_size': 1,
                    'shuffle': opt['validation']['shuffle'],
                    'drop_last': opt['validation']['drop_last'],
                    'collate_fn': collater,
                    'num_workers': opt['validation']['num_workers']}
    
    train_dataset = CustomDataset(image_path=opt['training']['image_path'], 
                        annotation_path=opt['training']['annotation_path'], 
                        image_size=opt['training']['image_size'], 
                        normalize=opt['training']['normalize'],
                        augmentations=augmentations)
    
    val_dataset = CustomDataset(image_path=opt['validation']['image_path'], 
                        annotation_path=opt['validation']['annotation_path'], 
                        image_size=opt['training']['image_size'], 
                        normalize=opt['training']['normalize'], 
                        augmentations=None)

    
    train_loader = DataLoader(train_dataset, **training_params)

    val_loader = DataLoader(val_dataset, **val_params)

    logger = TensorBoardLogger("tb_logs", name="my_model")

    loss_fn = FocalLoss()

    optimizer = get_optimizer(opt['training'], model)
    scheduler = get_scheduler(opt['training'], optimizer, len(train_loader))
    
    detector = Detector(model=model, scheduler=scheduler, optimizer=optimizer, loss=loss_fn)

    trainer = pl.Trainer(gpus=1, logger=logger, check_val_every_n_epoch=opt['training']['val_frequency'], max_epochs=opt['training']['epochs'])
    
    trainer.fit(model=detector, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)
                
