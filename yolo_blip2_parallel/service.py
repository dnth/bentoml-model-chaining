from __future__ import annotations

import asyncio
import json
from loguru import logger
import typing as t
from pathlib import Path

import bentoml
from bentoml.validators import ContentType
from PIL import Image as PILImage
import time

Image = t.Annotated[Path, ContentType("image/*")]


@bentoml.service()
class ObjectDetector:
    def __init__(self):
        import torch
        from ultralytics import YOLO

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Object detector using device: {device}")
        self.model = YOLO("yolov8x.pt").to(device=device)

    @bentoml.api(batchable=True)
    async def predict(self, images: list[Image]) -> list[list[dict]]:
        results = self.model.predict(source=images)
        return [json.loads(result.tojson()) for result in results]

    @bentoml.api
    def render(self, image: Image) -> Image:
        result = self.model.predict(image)[0]
        output = image.parent.joinpath(f"{image.stem}_result{image.suffix}")
        result.save(str(output))
        return output
    

@bentoml.service()
class Captioner:
    def __init__(self) -> None:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Captioner using device: {device}")
        self.pipeline = pipeline("image-to-text", model="Salesforce/blip2-opt-2.7b", device=device)

    @bentoml.api(batchable=True)
    async def caption(self, image: list[Image]) -> list[str]:
        images = [PILImage.open(img) for img in image]
        results = self.pipeline(images)
        return [item["generated_text"] for sublist in results for item in sublist]


@bentoml.service(resources={"gpu": 1})
class ParallelBLIP2YOLO:
    object_detection = bentoml.depends(ObjectDetector)
    captioning = bentoml.depends(Captioner)

    @bentoml.api()
    async def parallel_inference(self, images: list[Image]) -> t.Tuple[list[list[dict]], list[str], float]:
        start_time = time.time()

        object_detection, caption = await asyncio.gather(
            self.object_detection.predict(images),
            self.captioning.caption(images),
        )

        end_time = time.time()
        runtime = end_time - start_time
        logger.info(f"Parallel inference took {runtime:.2f} seconds")

        return object_detection, caption, runtime
