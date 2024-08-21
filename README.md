# bentoml-model-chaining
Chain multiple models together using BentoML. Deploy anywhere Docker runs.


## Run Object Detector
Models:
- YOLOv8

```bash
cd yolo/
bentoml serve .
```

## Run Image Captioner
Models:
- BLIP2

```bash
cd blip2
bentoml serve .
```

## Run Image Tagger
Models:
- RAM

```bash
cd ram
bentoml serve .
```
## Run Image Classification
Models:
- ResNet50

## Run Zero-shot Classification
Models:
- CLIP

## Run Zero-shot Detection

Models:
- Grounding DINO
- OWLv2

## Run Zero-shot Segmentation
Models:
- SAM
- SAM2

## Run Chaining Pipeline
Consists of running Image Tagger/Captioner -> Zero-shot Detection -> Segmentation


## Build and Push to Deploy
Build images and deploy to cloud.