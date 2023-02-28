# TyNet, A Custom Object Detector with a Training Pipeline
  
  TyNet is a lightweight, powerful, and scalable CNN-based object detector. You can choose a backbone from various backbones, or you can change the model itself, and then train another model by using the training pipeline.


## Project Structure
```bash
project/
├── README.md
├── LICENSE
├── Dockerfile
├── train.py
├── training.yaml
├── data/
│   ├── augmentations.py
│   ├── coco_dataset.py
│   ├── process_box.py
│   └── yolo_to_coco.py
├── datasets/
│   └── coco2017
└── models/
    ├── detector.py
    ├── loss.py
    ├── model.py
    ├── model_bn.py
    └── utils.py

```
## Getting Started
To get started with TyNet, follow these steps:

1- Clone this repository to your local machine.

```
git clone https://github.com/taylanates24/object-detection.git
```
2 - Build a docker image 

```
docker build -t tynet:v1 -f Dockerfile .
```

3- Create a container from docker image 

```
docker run -v $(pwd):/workspace -it --rm --ipc host tynet:v1
```

4 - Prepare your dataset in the COCO format.

5 - Modify the configuration file training.yaml to match your dataset, hyperparameters and data augmentations.

6 - Run 
```
python train.py
```
to start training the model.

## Data Preprocessing

TyNet uses the COCO dataset format for annotations. The data/coco_dataset.py script loads the images and annotations and preprocesses them for training.

### Augmentations

#### imgaug

I add some geometric and color augmentations from imgaug library. You can chance the number of augmentations in one time, the values and the variation of the augmentations by changing ```imgaug``` values in ```training.yaml``` file.

For example if you want to use random horizontal flip and scale only, the num_aug should be 2, fliplr should be the probability if the augmentation, scale should be a list of range and all the other imgaug augmentations should be ```null```

#### CutOut

It is my implementation of "Improved Regularization of Convolutional Neural Networks with Cutout" (https://arxiv.org/abs/1708.04552) paper with some improvements. In my implementation, you can change the filled boxes which cutted out. You can fill it with gaussian noise, random colors, and white, black, and gray boxes to the cutting area. You can also change the number of cutouts and their scale with respect to the height of the image by changing `cutout` `percentages` values in the `training.yaml` file.

#### Copy Paste

It is my implementation of "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" (https://arxiv.org/abs/2012.07177) paper with some improvements. In this implementation, you can aplly bounding box augmentations by changing copy_paste: box_augments: variable in ```training.yaml``` file. The bounding box augmentations only applied to the number of pasted boxes and at most 1 augmentation is applied at one time. You can also change the pasted boxes by changing `pasted_bbox_number` value in `training.yaml`

## Model Architecture

## Training

## Testing and Evaluation

## Conclusion and Future Work
