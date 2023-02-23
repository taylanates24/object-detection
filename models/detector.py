import pytorch_lightning as pl
import torch.nn as nn
import torch
from loss import FocalLoss

class Detector(pl.LightningModule):
    
    def __init__(self, model, scheduler, optimizer, loss=None):
        super(Detector, self).__init__()
        
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer

        if loss is None:
            self.loss = FocalLoss()
        else:
            self.loss = loss
            
    def forward(self, images, labels):
        
        return self.model(x)

    def training_step(self, train_batch, batch_idx):

        images = data['img']
        labels = data['labels']

        output = self.forward(images, labels)
        cls_loss, reg_loss = self.loss(classification, regression, anchors, annotations)

        self.log('learning rate', self.scheduler.get_lr()[0])

        return {'cls_loss': cls_loss, 'reg_loss': reg_loss}

    def training_epoch_end(self, outputs):
        
        cls_losses = [x['cls_loss'] for x in outputs]
        reg_losses = [x['reg_loss'] for x in outputs]
        
        avg_train_cls_loss = sum(cls_losses) / len(cls_losses)
        avg_train_reg_loss = sum(reg_losses) / len(reg_losses)

        self.log('train cls_loss', avg_train_cls_loss)
        self.log('train reg_loss', avg_train_reg_loss)

    def validation_step(self, val_batch, batch_idx):

        pass

    def validation_epoch_end(self, outputs):

        pass

    def test_step(self, batch, batch_idx):

        pass


    def configure_optimizers(self):

        pass