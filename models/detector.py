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
        self.best_val_loss = float('inf')

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
        total_loss = cls_loss + reg_loss

        self.log('learning rate', self.scheduler.get_lr()[0])

        return {'loss': total_loss, 'cls_loss': cls_loss, 'reg_loss': reg_loss}

    def training_epoch_end(self, outputs):
        
        cls_losses = [x['cls_loss'] for x in outputs]
        reg_losses = [x['reg_loss'] for x in outputs]
        total_losses = [x['loss'] for x in outputs]
        
        avg_train_cls_loss = sum(cls_losses) / len(cls_losses)
        avg_train_reg_loss = sum(reg_losses) / len(reg_losses)
        avg_train_loss = sum(total_losses) / len(total_losses)

        self.log('train cls_loss', avg_train_cls_loss)
        self.log('train reg_loss', avg_train_reg_loss)
        self.log('train total_loss', avg_train_loss)

    def validation_step(self, val_batch, batch_idx):

        images = data['img']
        labels = data['labels']

        output = self.forward(images, labels)
        cls_loss, reg_loss = self.loss(classification, regression, anchors, annotations)
        total_loss = cls_loss + reg_loss

        return {'loss':total_loss,  'cls_loss': cls_loss, 'reg_loss': reg_loss}

    def validation_epoch_end(self, outputs):

        cls_losses = [x['cls_loss'] for x in outputs]
        reg_losses = [x['reg_loss'] for x in outputs]
        total_losses = [x['loss'] for x in outputs]
        
        avg_val_cls_loss = sum(cls_losses) / len(cls_losses)
        avg_val_reg_loss = sum(reg_losses) / len(reg_losses)
        avg_val_loss = sum(total_losses) / len(total_losses)
        
        if  self.best_val_loss > avg_val_loss:
            self.best_val_loss = avg_val_loss

            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss,
                }, './best.ckpt')

        self.log('validation cls_loss', avg_val_cls_loss)
        self.log('validation reg_loss', avg_val_reg_loss)
        self.log('validation total_loss', avg_val_loss)
        

    def test_step(self, batch, batch_idx):

        pass


    def configure_optimizers(self):

        pass