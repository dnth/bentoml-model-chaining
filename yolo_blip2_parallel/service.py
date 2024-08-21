from __future__ import annotations

import asyncio
import json
import os
import typing as t
from pathlib import Path

import bentoml
from bentoml.validators import ContentType
from PIL import Image as PILImage

Image = t.Annotated[Path, ContentType("image/*")]


@bentoml.service()
class YoloV8:
    def __init__(self):
        import torch
        from ultralytics import YOLO

        yolo_model = os.getenv("YOLO_MODEL", "yolov8x.pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("#####")
        print(f"Using device: {device}")
        print("#####")
        self.model = YOLO(yolo_model).to(device=device)

    @bentoml.api(batchable=True)
    async def predict(self, images: list[Image]) -> list[list[dict]]:
        results = self.model.predict(source=images)
        # print("#####")
        # print(results)
        # print("#####")
        return [json.loads(result.tojson()) for result in results]

    @bentoml.api
    def render(self, image: Image) -> Image:
        result = self.model.predict(image)[0]
        output = image.parent.joinpath(f"{image.stem}_result{image.suffix}")
        result.save(str(output))
        return output
    

@bentoml.service()
class Captioning:
    def __init__(self) -> None:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # print("#####")
        # print(f"Using device: {device}")
        # print("#####")
        self.pipeline = pipeline("image-to-text", model="Salesforce/blip2-opt-2.7b", device=device)

    @bentoml.api(batchable=True)
    async def caption(self, image: list[Image]) -> list[str]:
        images = [PILImage.open(img) for img in image]
        results = self.pipeline(images)
        # print("#####")
        # print(results)
        # print("#####")
        return [item["generated_text"] for sublist in results for item in sublist]


@bentoml.service(resources={"gpu": 1})
class ParallelBLIP2YOLO:
    object_detection = bentoml.depends(YoloV8)
    captioning = bentoml.depends(Captioning)

    # @bentoml.api()
    # def sequential_inference(self, images: list[Image]) -> t.Tuple[list[list[dict]], list[str]]:
    #     object_detection = self.object_detection.predict(images)
    #     caption = self.captioning.caption(images)

    #     return object_detection, caption
    
    @bentoml.api()
    async def parallel_inference(self, images: list[Image]) -> t.Tuple[list[list[dict]], list[str]]:
        object_detection, caption = await asyncio.gather(
            self.object_detection.predict(images),
            self.captioning.caption(images),
        )

        return object_detection, caption
