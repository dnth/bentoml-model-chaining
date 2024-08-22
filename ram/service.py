from __future__ import annotations

import typing as t
from pathlib import Path

import bentoml
from bentoml.validators import ContentType
from PIL import Image as PILImage

import os
import requests

from loguru import logger
from tqdm.auto import tqdm

Image = t.Annotated[Path, ContentType("image/*")]


@bentoml.service()
class ImageTagging:
    def __init__(self) -> None:
        import torch
        from ram import get_transform
        from ram import inference_ram as inference
        from ram.models import ram_plus

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_transform(image_size=384)
        self.inference = inference

        # Prepare model
        pretrained_dir = "pretrained"
        model_filename = "ram_plus_swin_large_14m.pth"
        model_path = os.path.join(pretrained_dir, model_filename)
        model_url = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"

        if not os.path.isfile(model_path):
            logger.info(f"Pre-trained model not found at {model_path}. Downloading...")
            self.download_model(model_url, model_path)
        

        model = ram_plus(
            pretrained="pretrained/ram_plus_swin_large_14m.pth",
            image_size=384,
            vit="swin_l",
        )
        model.eval()
        self.model = model.to(self.device)
    
    def download_model(self, url, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get the total file size
        total_size = int(response.headers.get('content-length', 0))

        # Open the file and initialize the progress bar
        with open(path, 'wb') as file, tqdm(
            desc=path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
        
        logger.info(f"Model downloaded successfully to {path}")

    @bentoml.api()
    def tag(self, image: Image) -> dict:
        image = self.transform(PILImage.open(image)).unsqueeze(0).to(self.device)
        english_tags, chinese_tags = self.inference(image, self.model)
        english_tags = [tag.strip() for tag in english_tags.split("|")]
        chinese_tags = [tag.strip() for tag in chinese_tags.split("|")]
        return {"english_tags": english_tags, "chinese_tags": chinese_tags}

    #TODO add batch inference

    @bentoml.api()
    def tag_image_url(self, image_url: str) -> dict:
        import requests
        from io import BytesIO

        response = requests.get(image_url)
        image = PILImage.open(BytesIO(response.content))
        image = self.transform(image).unsqueeze(0).to(self.device)
        english_tags, chinese_tags = self.inference(image, self.model)
        english_tags = [tag.strip() for tag in english_tags.split("|")]
        chinese_tags = [tag.strip() for tag in chinese_tags.split("|")]
        return {"english_tags": english_tags, "chinese_tags": chinese_tags}