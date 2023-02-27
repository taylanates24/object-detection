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

## Data Preprocessing

## Model Architecture

## Training

## Testing and Evaluation

## Conclusion and Future Work
