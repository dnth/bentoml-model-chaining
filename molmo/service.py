import typing as t
from pathlib import Path

import bentoml
import requests
import torch
from bentoml.validators import ContentType
from loguru import logger
from PIL import Image as PILImage
from vllm import LLM, SamplingParams

Image = t.Annotated[Path, ContentType("image/*")]

# Download model from huggingface into local directory
# huggingface-cli download allenai/Molmo-7B-D-0924 --local-dir molmo_7b_d_0924


@bentoml.service()
class MolmoService:
    def __init__(self):
        model_id = "molmo_7b_d_0924/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.llm = LLM(
            model=model_id,
            trust_remote_code=True,
            dtype="bfloat16",
        )

    @bentoml.api()
    def caption_image_url(self, image_url: str) -> str:
        prompt = "Describe this image."
        sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

        image = PILImage.open(requests.get(image_url, stream=True).raw).convert("RGB")

        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }

        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        return generated_text

    @bentoml.api(batchable=True, max_batch_size=10)
    def caption_image_urls(self, image_urls: list[str]) -> list[str]:
        prompt = "Describe this image."
        sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

        batch_inputs = [
            {
                "prompt": f"USER: <image>\n{prompt}\nASSISTANT:",
                "multi_modal_data": {
                    "image": PILImage.open(
                        requests.get(image_url, stream=True).raw
                    ).convert("RGB")
                },
            }
            for image_url in image_urls
        ]

        outputs = self.llm.generate(batch_inputs, sampling_params=sampling_params)
        return [output.outputs[0].text for output in outputs]
