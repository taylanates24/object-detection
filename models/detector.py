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
            
    def forward(self, x):
        
        return self.model(x)

    def training_step(self, train_batch, batch_idx):

        pass

    def training_epoch_end(self, outputs):

        pass

    def validation_step(self, val_batch, batch_idx):

        pass

    def validation_epoch_end(self, outputs):

        pass

    def test_step(self, batch, batch_idx):

        pass


    def configure_optimizers(self):

        pass