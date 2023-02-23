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