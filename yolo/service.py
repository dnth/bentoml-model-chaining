from __future__ import annotations

import json
import typing as t
from pathlib import Path

import bentoml
from bentoml.validators import ContentType

Image = t.Annotated[Path, ContentType("image/*")]


@bentoml.service()
class YoloV8:
    def __init__(self):
        from ultralytics import YOLO

        self.model = YOLO("yolov8s.pt")

    @bentoml.api(batchable=True)
    def detect_files(self, images: list[Image]) -> list[list[dict]]:
        results = self.model.predict(source=images)
        return [json.loads(result.tojson()) for result in results]

    @bentoml.api(batchable=True)
    def detect_urls(self, urls: list[str]) -> list[list[dict]]:
        results = self.model.predict(source=urls)
        return [json.loads(result.tojson()) for result in results]

    @bentoml.api
    def render_detect_file(self, image: Image) -> Image:
        result = self.model.predict(image)[0]
        output = image.parent.joinpath(f"{image.stem}_result{image.suffix}")
        result.save(str(output))
        return output

    @bentoml.api
    def render_detect_url(self, url: str) -> Image:
        result = self.model.predict(url)[0]
        output_filename = f"result_{hash(url)}.jpg"
        output = Path(output_filename)
        result.save(str(output))
        return output
