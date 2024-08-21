from __future__ import annotations

import typing as t
from pathlib import Path

import bentoml
from bentoml.validators import ContentType
from PIL import Image as PILImage

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

        model = ram_plus(
            pretrained="pretrained/ram_plus_swin_large_14m.pth",
            image_size=384,
            vit="swin_l",
        )
        model.eval()
        self.model = model.to(self.device)

    @bentoml.api()
    def tag(self, image: Image) -> dict:
        image = self.transform(PILImage.open("demo1.jpg")).unsqueeze(0).to(self.device)
        english_tags, chinese_tags = self.inference(image, self.model)
        english_tags = [tag.strip() for tag in english_tags.split("|")]
        chinese_tags = [tag.strip() for tag in chinese_tags.split("|")]
        return {"english_tags": english_tags, "chinese_tags": chinese_tags}

    #TODO add batch inference