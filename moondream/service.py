from __future__ import annotations

import typing as t
from pathlib import Path

import bentoml
import requests
import torch
from bentoml.validators import ContentType
from loguru import logger
from PIL import Image as PILImage
from transformers import AutoModelForCausalLM, AutoTokenizer

Image = t.Annotated[Path, ContentType("image/*")]


@bentoml.service()
class MoondreamService:
    def __init__(self) -> None:
        model_id = "vikhyatk/moondream2"
        revision = "2024-08-26"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading model from {model_id} with revision {revision}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            cache_dir="./moondream2",
        ).to(self.device)

        self.model = torch.compile(self.model, mode="max-autotune")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def _run_inference(self, image: PILImage.Image, question: str) -> str:
        logger.info(f"Running inference on image with question: {question}")
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            enc_image = self.model.encode_image(image).to(self.device)
            return self.model.answer_question(enc_image, question, self.tokenizer)

    @bentoml.api()
    def vqa_image_url(self, image_url: str, question: str) -> str:
        image = PILImage.open(requests.get(image_url, stream=True).raw)
        return self._run_inference(image, question)

    @bentoml.api()
    def caption_image_url(self, image_url: str) -> str:
        image = PILImage.open(requests.get(image_url, stream=True).raw)
        prompt = "Describe this image in detail, focusing on the main subjects, \
                  their actions, and the overall setting. Include information about \
                  colors, textures, and any notable objects or elements in the background."
        return self._run_inference(image, prompt)
