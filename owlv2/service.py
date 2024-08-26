from __future__ import annotations

import bentoml
import typing as t
from pathlib import Path
from bentoml.validators import ContentType
from PIL import Image as PILImage
import colorsys

Image = t.Annotated[Path, ContentType("image/*")]

@bentoml.service(resources={"gpu": 1})
class OWLv2:
    def __init__(self) -> None:
        from transformers import pipeline
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = "google/owlv2-base-patch16-ensemble"
        self.detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device=self.device)

    @bentoml.api
    def detect_objects(self, image: Image, queries: str, confidence_threshold: float = 0.5) -> list[dict]:
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
    def detect_and_render(self, image: Image, queries: str, confidence_threshold: float = 0.5) -> Image:
        from PIL import ImageDraw, ImageFont
        

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