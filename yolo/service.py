from __future__ import annotations

import json
import time
import typing as t
from pathlib import Path

import bentoml
import torch
from bentoml.validators import ContentType
from loguru import logger

logger.add("bentoml.log", rotation="1 day", retention="10 days", compression="zip")

Image = t.Annotated[Path, ContentType("image/*")]


@bentoml.service()
class YoloV8:
    def __init__(self):
        from ultralytics import YOLO

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        self.model = YOLO("yolov8s.pt").to(device)

    def _log_inference_latency(self, start_time):
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        logger.debug(f"Inference latency: {latency:.2f} ms")
        if latency > 5000:  # Log a warning if inference takes more than 5 seconds
            logger.warning(f"High inference latency detected: {latency:.2f} ms")

    # Add this method to log inference latency for all API methods
    def _log_and_return(self, start_time, result):
        self._log_inference_latency(start_time)
        return result

    @bentoml.api(batchable=True, max_batch_size=20)
    def detect_files(self, images: list[Image]) -> list[list[dict]]:
        start_time = time.time()
        results = self.model.predict(source=images)
        return self._log_and_return(
            start_time, [json.loads(result.tojson()) for result in results]
        )

    @bentoml.api(batchable=True, max_batch_size=20)
    def detect_urls(self, urls: list[str]) -> list[list[dict]]:
        results = self.model.predict(
            source=urls
        )  # TODO: handle cases where url does not end with image extentions like jpg
        return [json.loads(result.tojson()) for result in results]

    @bentoml.api
    def detect_file(self, image: Image) -> list[dict]:
        result = self.model.predict(image)[0]
        return json.loads(result.tojson())

    @bentoml.api
    def detect_url(self, url: str) -> list[dict]:
        result = self.model.predict(url)[
            0
        ]  # TODO: handle cases where url does not end with image extentions like jpg
        return json.loads(result.tojson())

    @bentoml.api
    def render_detect_file(self, image: Image) -> Image:
        result = self.model.predict(image)[0]
        output = image.parent.joinpath(f"{image.stem}_result{image.suffix}")
        result.save(str(output))
        return output

    @bentoml.api
    def render_detect_url(self, url: str) -> Image:
        result = self.model.predict(url)[
            0
        ]  # TODO: handle cases where url does not end with image extentions like jpg
        output = Path("result.jpg")
        result.save(str(output))
        return output
