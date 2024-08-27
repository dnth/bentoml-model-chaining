from __future__ import annotations
import bentoml


from pathlib import Path
import typing as t
from bentoml.validators import ContentType
from PIL import Image as PILImage


Image = t.Annotated[Path, ContentType("image/*")]

@bentoml.service(resources={"gpu": 1})
class Captioning:
    def __init__(self) -> None:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        self.pipeline = pipeline("image-to-text", model="Salesforce/blip2-opt-2.7b", device=device)

    @bentoml.api(batchable=True)
    def caption(self, image: list[Image]) -> list[str]:
        images = [PILImage.open(img) for img in image]
        results = self.pipeline(images)
        print(results)
        return [item["generated_text"] for sublist in results for item in sublist]
    
    @bentoml.api()
    def caption_image_file(self, image: Image) -> str:
        return self.caption([image])[0]