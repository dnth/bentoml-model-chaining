from __future__ import annotations

import typing as t
from pathlib import Path

import bentoml
import requests
import torch
from bentoml.validators import ContentType
from PIL import Image as PILImage
from transformers import AutoModelForCausalLM, AutoProcessor

Image = t.Annotated[Path, ContentType("image/*")]


@bentoml.service()
class Phi35VisionService:
    def __init__(self) -> None:
        model_id = "./models-phi-35-vision"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, num_crops=16
        )

        self.model = torch.compile(self.model, mode="max-autotune")

    def _run_inference(
        self, image: PILImage.Image, prompt_text: str, max_new_tokens: int = 50
    ) -> str:
        # Note: Batch inference is not supported for this model
        placeholder = f"<|image_1|>\n"
        messages = [
            {"role": "user", "content": placeholder + prompt_text},
        ]
        prompt_text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            inputs = self.processor(prompt_text, image, return_tensors="pt").to(
                self.device
            )
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
            return self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

    @bentoml.api()
    def vqa_image_url(
        self, image_url: str, question: str, max_new_tokens: int = 50
    ) -> str:
        image = PILImage.open(requests.get(image_url, stream=True).raw)
        return self._run_inference(image, question, max_new_tokens)

    @bentoml.api()
    def caption_image_url(self, image_url: str, max_new_tokens: int = 50) -> str:
        image = PILImage.open(requests.get(image_url, stream=True).raw)
        prompt = "Describe the image in concise, focusing on the main subjects, \
                  their actions, and the overall setting. Include information about \
                  colors, textures, and any notable objects or elements in the background. \
                  Eliminate filler words, adverbs, and any unnecessary phrases, focusing solely \
                  on the core meaning and essential information."
        return self._run_inference(image, prompt, max_new_tokens)
