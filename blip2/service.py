from __future__ import annotations

import time
import typing as t
from io import BytesIO
from pathlib import Path

import bentoml
import requests
from bentoml.validators import ContentType
from loguru import logger
from PIL import Image as PILImage

logger.add("bentoml.log", rotation="1 day", retention="10 days", compression="zip")
Image = t.Annotated[Path, ContentType("image/*")]


@bentoml.service()
class Blip2Captioning:
    def __init__(self) -> None:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        logger.info("Loading model...")
        self.pipeline = pipeline(
            "image-to-text",
            model="Salesforce/blip2-opt-2.7b",
            device=device,
            torch_dtype=torch.float16,
        )
        logger.info("Model loaded successfully")

    def _log_inference_latency(self, start_time):
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        logger.debug(f"Inference latency: {latency:.2f} ms")
        if latency > 5000:  # Log a warning if inference takes more than 5 seconds
            logger.warning(f"High inference latency detected: {latency:.2f} ms")

    def run_inference(self, images: list[PILImage.Image]) -> list[str]:
        logger.debug(f"Starting inference for {len(images)} images")
        start_time = time.time()
        results = self.pipeline(images)
        self._log_inference_latency(start_time)
        logger.debug(f"Inference completed, generated {len(results)} captions")
        return [item["generated_text"] for sublist in results for item in sublist]

    @bentoml.api(batchable=True)
    def batch_caption_image_files(self, image: list[Image]) -> list[dict]:
        logger.info(f"Running batch inference for {len(image)} images")
        images = [PILImage.open(img) for img in image]
        captions = self.run_inference(images)
        return [{"caption": caption} for caption in captions]

    @bentoml.api()
    def caption_image_file(self, image: Image) -> dict:
        logger.info(f"Running inference for 1 image")
        caption = self.batch_caption_image_files([image])[0]["caption"]
        return {"caption": caption}

    @bentoml.api()
    def caption_image_url(self, image_url: str) -> dict:
        logger.info(f"Running inference for 1 image URL - {image_url}")
        response = requests.get(image_url)
        image = PILImage.open(BytesIO(response.content))
        caption = self.run_inference([image])[0]
        return {"caption": caption}

    @bentoml.api(batchable=True)
    def batch_caption_image_urls(self, image_urls: list[str]) -> list[dict]:
        logger.info(f"Running batch inference for {len(image_urls)} image URLs")
        responses = [requests.get(url) for url in image_urls]
        images = [PILImage.open(BytesIO(response.content)) for response in responses]
        captions = self.run_inference(images)
        return [{"caption": caption} for caption in captions]
