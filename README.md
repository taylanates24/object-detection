# TyNet, A Custom Object Detector with a Pytorch-lightning Training Pipeline
  
  TyNet is a lightweight, powerful, and scalable CNN-based object detector. You can choose a backbone from various backbones, or you can change the model itself, and then train another model by using the training pipeline.


## Project Structure
```bash
object-detection/
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
2 - Build a docker image and create a container from docker image  (recommended)

```
docker build -t tynet:v1 -f Dockerfile .
```

```
docker run -v $(pwd):/workspace -it --rm --ipc host tynet:v1
```

3 - (Optional) If you are using python virtual environment, create a virtual environment and install requirements.

```
python3 -m venv tynet
source tynet/bin/activate
pip3 install -r requirements.txt
```
4 - Prepare your dataset in the COCO format or if you want a small dataset to quickly build the environment, run the following code to download coco128 dataset. (the first 128 images and labels from coco2017 dataset)

```
chmod +x get_coco128.sh
./get_coco128.sh

```
After that, you have to convert the dataset format to COCO format by running the following code:

```
python3 data/yolo_to_coco.py --yolo_annotations /workspaces/object-detection/datasets/coco128/labels/train2017 --yolo_img /workspaces/object-detection/datasets/coco128/images/train2017 --coco_names /workspaces/object-detection/datasets/coco.names --out_file_name /workspaces/object-detection/datasets/coco128_train.json --check True
```

5 - Modify the configuration file training.yaml to match your dataset, hyperparameters and data augmentations.

6 - Run 
```
python3 train.py --train_cfg training.yaml --dataset_cfg coco.yml
```
to start training the model.

## Data Preprocessing

TyNet uses the COCO dataset format for annotations. The data/coco_dataset.py script loads the images and annotations and preprocesses them for training.

### Augmentations

#### imgaug

I add some geometric and color augmentations from imgaug library. You can chance the number of augmentations in each iteration, the values and the variation of the augmentations by changing ```imgaug``` values in ```training.yaml``` file.

For example if you want to use random horizontal flip and scale only, the num_aug should be 2, fliplr should be the probability if the random horizontal flip augmentation, scale should be a list of range and all the other imgaug augmentations should be ```null``` The `num_aug` is the number of augmentations, which is chosen randomly from all augmentations,  on each iteration.

#### CutOut

It is my implementation of "Improved Regularization of Convolutional Neural Networks with Cutout" (https://arxiv.org/abs/1708.04552) paper with some improvements. In my implementation, you can change the filled boxes which cutted out. You can fill it with gaussian noise, random colors, and white, black, and gray boxes to the cutting area. You can also change the number of cutouts and their scale with respect to the height of the image by changing `cutout` `percentages` values in the `training.yaml` file. The lenght of `percentages` is the number of cutting boxes.

#### Copy Paste

It is my implementation of "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" (https://arxiv.org/abs/2012.07177) paper with some improvements. In this implementation, you can apply bounding box augmentations by changing copy_paste: box_augments: variable in ```training.yaml``` file. The bounding box augmentations only applied to the number of pasted boxes and at most 1 augmentation is applied at one time. You can also change the pasted boxes by changing `pasted_bbox_number` value in `training.yaml`

## Model Architecture

### Backbone

 As a backbone, I implement a changable structure. You can use whatever backbone in the `timm` library, pre-trained or not. For example, the list of pre-trained backbones can be obtained by using the folloving code:
 
```
import timm

available_backbones = timm.list_models(pretrained=True)
```

### Neck
As a neck, I designed an FPN structure, which uses addition as fusion type. 

![fpn_arch (copy) drawio](https://user-images.githubusercontent.com/66252663/221975984-fec13a49-b66b-4f35-b837-78278b434868.png)

Additionally, I desing a simple but effective and computationally cheap, scaleble block: ScalableCSPResBlock. It is scalable, and designed using CSPNet on top of Resnet.

![scalablecspresblock (1)](https://user-images.githubusercontent.com/66252663/221976374-8571cdd7-8062-4966-8d93-6facdb350718.jpg)


### Head

This is my implementation of bounding box regression head. Each `Conv2d` consists of one 3x3 Conv2d and one 1x1 Conv2d. At the end, there will be 1xNx4 dimensional tensor, N proposal coordinates for each bbox coordinate.

![classification](https://user-images.githubusercontent.com/66252663/221974242-17bdcc6a-6c83-40b2-b578-d7d37edcb3dd.jpg)

This is my implementation of bounding box classification head. Similarly, each `Conv2d` consists of one 3x3 Conv2d and one 1x1 Conv2d. At the end, there will be 1xNx80 dimensional tensor, N class probabilities for each class in COCO dataset.

![classification_corrected](https://user-images.githubusercontent.com/66252663/221975023-3c9aa045-0d61-4ec0-a32c-8a91801f6426.jpg)

## Training

To train the TyNet, you can edit the training pipeline such as learning rate scheduler, optimizer and so on.

### Optimizer

 I have implemented 4 optimizers in this repository, Adam, AdamW, SGD and ASGD. You can choose one of them by changing `training: optimizer:` in `training.yaml.` file.
 
### Learning Rate Scheduler

There are 3 learning rate schedulers in this repository, cosine, multistep_lr and cosine_annealing. You can choose one of them by changing `training: lr_scheduler:` in `training.yaml.` file.

### Validation

There is also a validation frequency that indicates the time when the model validates. It is set to 1, that means every epoch, the validation phase occurs.

## Testing and Evaluation

Coming soon

## Conclusion and Future Work

The head part of the model will be written by using CSPNet to have a more accurate and fast head part.
