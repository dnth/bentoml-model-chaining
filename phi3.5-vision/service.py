from __future__ import annotations
import bentoml
from pathlib import Path
import typing as t
from bentoml.validators import ContentType
from PIL import Image as PILImage
import requests
from transformers import AutoModelForCausalLM, AutoProcessor

Image = t.Annotated[Path, ContentType("image/*")]

@bentoml.service(resources={"gpu": 1})
class Phi35VisionService:
    def __init__(self) -> None:
        model_id = "microsoft/Phi-3.5-vision-instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation='flash_attention_2'
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=4
        )

    def _process_images(self, images: list[Image | str]) -> list[PILImage.Image]:
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(PILImage.open(requests.get(img, stream=True).raw))
            else:
                pil_images.append(PILImage.open(img))
        return pil_images

    def _generate_response(self, images: list[PILImage.Image], prompt: str) -> str:
        placeholder = "".join([f"<|image_{i+1}|>\n" for i in range(len(images))])
        messages = [
            {"role": "user", "content": placeholder + prompt},
        ]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda:0")
        generation_args = {
            "max_new_tokens": 1000,
            "temperature": 0.0,
            "do_sample": False,
        }
        generate_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        return self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    @bentoml.api(batchable=True)
    def analyze(self, images: list[Image | str], prompt: str) -> str:
        pil_images = self._process_images(images)
        return self._generate_response(pil_images, prompt)

    @bentoml.api()
    def ocr(self, image: Image) -> str:
        return self.analyze([image], "Perform OCR on this image and extract all the text you can see.")

    @bentoml.api()
    def chart_analysis(self, image: Image) -> str:
        return self.analyze([image], "Analyze this chart or table. Describe its type, main components, and key insights.")

    @bentoml.api()
    def compare_images(self, images: list[Image]) -> str:
        return self.analyze(images, "Compare these images and describe the similarities and differences.")

    @bentoml.api()
    def summarize_sequence(self, images: list[Image]) -> str:
        return self.analyze(images, "Summarize the sequence of images as if they were frames from a video clip.")
    
    @bentoml.api()
    def caption_image(self, image: Image, prompt: str) -> str:
        # Process the single image
        pil_image = self._process_images([image])[0]
        
        # Generate the caption
        return self._generate_response([pil_image], prompt)
