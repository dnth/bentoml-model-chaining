from __future__ import annotations
import asyncio
import typing as t
from pathlib import Path
import bentoml
from bentoml.validators import ContentType
from PIL import Image as PILImage
import requests
from transformers import AutoModelForCausalLM, AutoProcessor
from loguru import logger
import colorsys
from textblob import TextBlob
from PIL import ImageDraw, ImageFont

Image = t.Annotated[Path, ContentType("image/*")]

@bentoml.service(resources={"gpu": 1})
class Phi35Vision:
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
            num_crops=16
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
        
    @bentoml.api()
    def caption_image(self, image: Image) -> t.Any: # not sure why str return does not work
        prompt = "Describe the image in detail."
        pil_images = self._process_images([image])
        caption = self._generate_response(pil_images, prompt)
        return caption

@bentoml.service()
class OWLv2:
    def __init__(self) -> None:
        from transformers import pipeline
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = "google/owlv2-base-patch16-ensemble"
        self.detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device=self.device)

    @bentoml.api
    def detect_objects(self, image: Image, queries: str, confidence_threshold: float = 0.3) -> list[dict]:
        image = PILImage.open(image).convert("RGB")
        
        # Split the queries string into a list
        query_list = [q.strip() for q in queries.split(',')]
        
        results = self.detector(image, candidate_labels=query_list)
        
        detections = []
        for result in results:
            if result["score"] >= confidence_threshold:
                detections.append({
                    "score": result["score"],
                    "label": result["label"],
                    "box": result["box"]
                })

        return detections
    
    @bentoml.api
    def detect_and_render(self, image: Image, queries: str, confidence_threshold: float = 0.3) -> t.Any:
        detections = self.detect_objects(image, queries, confidence_threshold)
        img = PILImage.open(image).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

        # Generate distinct colors for each unique label
        unique_labels = set(detection["label"] for detection in detections)
        colors = {label: self._get_distinct_color(i, len(unique_labels)) for i, label in enumerate(unique_labels)}

        # draw bounding boxes and labels
        for detection in detections:
            box = detection["box"]
            label = detection["label"]
            score = detection["score"]
            color = colors[label]
            
            # Ensure box is a sequence of 4 values
            if isinstance(box, dict):
                box = (box['xmin'], box['ymin'], box['xmax'], box['ymax'])
            
            # Draw rectangle
            draw.rectangle(box, outline=color, width=2)
            
            # Prepare label text
            label_text = f"{label}: {score:.2f}"
            
            # Get text bounding box
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Calculate position for text
            text_position = (box[0], box[1] - text_height - 5)
            
            # Draw background rectangle for text
            draw.rectangle([text_position[0], text_position[1], 
                            text_position[0] + text_width, text_position[1] + text_height], 
                           fill=color)
            
            # Draw text
            draw.text(text_position, label_text, fill="white", font=font)
        
        return img

    def _get_distinct_color(self, i: int, total: int) -> str:
        hue = i / total
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

@bentoml.service()
class InferenceGraph:
    phi35_vision = bentoml.depends(Phi35Vision)
    owlv2_detector = bentoml.depends(OWLv2)

    @bentoml.api()
    async def caption_and_detect(
        self, 
        image: Image, 
        confidence_threshold: float = 0.3
    ) -> dict:
        # Step 1: Generate caption using Phi 3.5 Vision
        caption = await self.phi35_vision.to_async.caption_image(image)

        logger.info(f"Caption: {caption}")

        # Get nouns from caption
        blob = TextBlob(caption)
        nouns = list(blob.noun_phrases)
        nouns = ", ".join(nouns)
        logger.info(f"Nouns: {nouns}")

        # Step 2: Use the generated caption for OWLv2 detection
        detections = await self.owlv2_detector.to_async.detect_objects(image, nouns)

        logger.info(f"Detections: {detections}")

        # Filter detections based on confidence threshold
        filtered_detections = [
            detection for detection in detections 
            if detection["score"] >= confidence_threshold
        ]

        logger.info(f"Filtered detections: {filtered_detections}")

        return {
            "caption": caption,
            "detections": filtered_detections
        }
    
    @bentoml.api()
    async def caption_detect_render(self, image: Image, confidence_threshold: float = 0.3) -> t.Any:
        caption = await self.phi35_vision.to_async.caption_image(image)

        logger.info(f"Caption: {caption}")

        blob = TextBlob(caption)
        nouns = list(blob.noun_phrases)
        nouns = ", ".join(nouns)
        logger.info(f"Nouns: {nouns}")

        rendered_image = await self.owlv2_detector.to_async.detect_and_render(image, nouns)

        return rendered_image
