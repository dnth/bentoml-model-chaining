from __future__ import annotations

import time
import typing as t
from io import BytesIO
from pathlib import Path

import bentoml
import requests
import torch
from bentoml.validators import ContentType
from loguru import logger
from PIL import Image as PILImage

logger.add("bentoml.log", rotation="1 day", retention="10 days", compression="zip")

Image = t.Annotated[Path, ContentType("image/*")]

@bentoml.service()
class RecognizeAnythingModel:
    def __init__(self) -> None:
        from ram import get_transform
        from ram import inference_ram as inference
        from ram.models import ram

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.transform = get_transform(image_size=384)
        self.inference = inference

        logger.info("Loading model...")
        model_path = "ram_swin_large_14m.pth"

        model = ram(
            pretrained=model_path,
            image_size=384,
            vit="swin_l",
        )
        model.eval()
        self.model = model.to(self.device)

        logger.info("Model loaded successfully")

    def _log_inference_latency(self, start_time):
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        logger.debug(f"Inference latency: {latency:.2f} ms")
        if latency > 5000:  # Log a warning if inference takes more than 5 seconds
            logger.warning(f"High inference latency detected: {latency:.2f} ms")

    @bentoml.api()
    def tag_image_file(self, image: Image) -> dict:
        logger.info(f"Running inference for 1 image file")
        image = self.transform(PILImage.open(image)).unsqueeze(0).to(self.device)
        start_time = time.time()
        with torch.inference_mode():
            english_tags, _ = self.inference(image, self.model)
        self._log_inference_latency(start_time)
        english_tags = [tag.strip() for tag in english_tags.split("|")]
        return {"english_tags": english_tags}

    @bentoml.api()
    def tag_image_url(self, image_url: str) -> dict:
        logger.info(f"Running inference for 1 image URL - {image_url}")
        response = requests.get(image_url)
        image = PILImage.open(BytesIO(response.content))
        image = self.transform(image).unsqueeze(0).to(self.device)
        start_time = time.time()
        with torch.inference_mode():
            english_tags, _ = self.inference(image, self.model)
        self._log_inference_latency(start_time)
        english_tags = [tag.strip() for tag in english_tags.split("|")]
        return {"english_tags": english_tags}

    @bentoml.api(batchable=True, max_batch_size=20)
    def batch_tag_image_urls(self, image_urls: list[str]) -> list[dict]:
        logger.info(f"Running batch inference for {len(image_urls)} image URLs")
        results = []
        for image_url in image_urls:
            tags = self.tag_image_url(image_url) # TOOD: Model is not doing batch inference
            results.append({"english_tags": tags["english_tags"]})
        return results

    @bentoml.api(batchable=True, max_batch_size=20)
    def batch_tag_image_files(self, images: list[Image]) -> list[dict]:
        logger.info(f"Running batch inference for {len(images)} images")
        results = []
        for image in images:
            tags = self.tag_image_file(image) # TOOD: Model is not doing batch inference
            results.append({"english_tags": tags["english_tags"]})
        return results
