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

## Model Architecture

## Training

## Testing and Evaluation

## Conclusion and Future Work
