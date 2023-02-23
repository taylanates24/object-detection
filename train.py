import pytorch_lightning as pl
import yaml
from models.model import TyNet
from models.utils import init_weights
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', type=str, default='train.yaml', help='training config file')
    parser.add_argument('--dataset_cfg', type=str, default='train.yaml', help='training config file')
    
    args = parser.parse_args()

    opt = args.train_cfg
    with open(opt, 'r') as config:
        opt = yaml.safe_load(config)

    dataset_opt = args.dataset_cfg
    with open(opt, 'r') as config:
        dataset_opt = yaml.safe_load(config)


    model = TyNet(compound_coef=0, num_classes=len(dataset_opt.obj_list),
                                    ratios=eval(dataset_opt.anchors_ratios), scales=eval(dataset_opt.anchors_scales))



    init_weights(model)