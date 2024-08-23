# bentoml-model-chaining
Chain multiple models together using BentoML. Deploy anywhere Docker runs.


## Object Detector
Models:
- YOLOv8

```bash
cd yolo/
bentoml serve .
```

## Image Captioner
Models:
- BLIP2

```bash
cd blip2
bentoml serve .
```

## Image Tagger
Models:
- RAM
- RAM++

```bash
cd ram
bentoml serve .
```
## Image Classification
Models:
- ResNet50

## Optical Character Recognition
Models:
- EasyOCR

## Compute Embeddings
Models:
- CLIP
- Sentence Transformers

## Zero-shot Classification
Models:
- CLIP

## Zero-shot Detection

Models:
- Grounding DINO
- OWLv2

## Zero-shot Segmentation
Models:
- SAM
- SAM2

## Run Pipeline
Consists of running Image Tagger/Captioner -> Zero-shot Detection -> Segmentation


## Build and Push to Deploy
Build images and deploy to cloud.