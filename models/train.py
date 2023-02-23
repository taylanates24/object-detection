import pytorch_lightning as pl





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', type=str, default='train.yaml', help='training config file')
    
    args = parser.parse_args()

    