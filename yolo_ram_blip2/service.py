import json
import os
import time
from io import BytesIO

import bentoml
import PIL
import requests
import torch
from loguru import logger
from ram import get_transform
from ram import inference_ram as inference
from ram.models import ram
from transformers import pipeline
from ultralytics import YOLO

logger.add("bentoml.log", rotation="1 day", retention="10 days", compression="zip")


class YOLOv8:
    def __init__(self):
        logger.info("Loading YOLOv8 model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        model_path = "yolov8s.pt"
        if not os.path.exists(model_path):
            logger.warning(f"Model file {model_path} not found. Downloading...")

        try:
            self.model = YOLO(model_path).to(self.device)
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {str(e)}")
            raise


class RecognizeAnythingModel:
    def __init__(self):
        logger.info("Loading RAM model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.transform = get_transform(image_size=384)
        self.inference = inference

        model_path = "ram_swin_large_14m.pth"
        if not os.path.exists(model_path):
            logger.error(
                f"Model file {model_path} not found. Please download it first."
            )
            raise FileNotFoundError(f"Model file {model_path} not found")

        try:
            self.model = ram(
                pretrained=model_path,
                image_size=384,
                vit="swin_l",
            )
            self.model.eval()
            self.model = self.model.to(self.device)
            logger.info(
                f"RAM model loaded successfully. Model type: {type(self.model).__name__}"
            )
        except Exception as e:
            logger.error(f"Error loading RAM model: {str(e)}")
            raise


class Blip2:
    def __init__(self) -> None:
        logger.info("Loading BLIP-2 model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        try:
            self.model = pipeline(
                "image-to-text",
                model="blip2-fp16",
                device=self.device,
                torch_dtype=torch.float16,
            )

            # self.model.save_pretrained("blip2-fp16/")
            logger.info(f"BLIP-2 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BLIP-2 model: {str(e)}")
            raise


@bentoml.service()
class EnrichmentModels:
    def __init__(self):
        self.yolov8 = YOLOv8()
        self.ram = RecognizeAnythingModel()
        self.blip2 = Blip2()

    def _log_inference_latency(self, start_time):
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        logger.debug(f"Inference latency: {latency:.2f} ms")
        if latency > 5000:  # Log a warning if inference takes more than 5 seconds
            logger.warning(f"High inference latency detected: {latency:.2f} ms")

    @bentoml.api(batchable=True, max_batch_size=20)
    def detect_urls(self, urls: list[str]) -> list[list[dict]]:
        start_time = time.time()
        results = self.yolov8.model.predict(source=urls, imgsz=640)
        self._log_inference_latency(start_time)
        return [json.loads(result.tojson()) for result in results]

    @bentoml.api(batchable=True, max_batch_size=20)
    def tag_urls(self, urls: list[str]) -> list[list[str]]:
        logger.info(f"Running batch inference for {len(urls)} image URLs")
        start_time = time.time()
        results = []
        for url in urls:
            response = requests.get(url)
            image = PIL.Image.open(BytesIO(response.content))
            image = self.ram.transform(image).unsqueeze(0).to(self.ram.device)
            with torch.inference_mode():
                english_tags, _ = self.ram.inference(image, self.ram.model)
            english_tags = [tag.strip() for tag in english_tags.split("|")]
            results.append(english_tags)
        self._log_inference_latency(start_time)
        return results

    @bentoml.api(batchable=True, max_batch_size=20)
    def caption_urls(self, image_urls: list[str]) -> list[str]:
        logger.info(f"Running batch inference for {len(image_urls)} image URLs")
        start_time = time.time()
        responses = [requests.get(url) for url in image_urls]
        images = [PIL.Image.open(BytesIO(response.content)) for response in responses]
        captions = self.blip2.model(images)
        self._log_inference_latency(start_time)
        return [caption[0]["generated_text"] for caption in captions]
