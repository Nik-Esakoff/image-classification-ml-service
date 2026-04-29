import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from ml.tiny_imagenet_resnet import ImageNetNN


MODEL_DIR = Path("models/tiny_imagenet_resnet")
MODEL_PATH = MODEL_DIR / "checkpoint.pth"
IDX_TO_LABEL_PATH = MODEL_DIR / "idx_to_label.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None


@lru_cache(maxsize=1)
def load_idx_to_label() -> dict[int, str]:
    if not IDX_TO_LABEL_PATH.exists():
        return {}

    data = json.loads(IDX_TO_LABEL_PATH.read_text(encoding="utf-8"))

    return {
        int(index): label
        for index, label in data.items()
    }


def format_prediction_item(class_index: int, probability: float) -> dict[str, Any]:
    idx_to_label = load_idx_to_label()
    label = idx_to_label.get(class_index, str(class_index))

    return {
        "class_index": class_index,
        "class_name": label,
        "label": label,
        "probability": round(float(probability), 6),
    }


def load_model_once():
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = ImageNetNN(num_classes=200).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    _model = model

    return _model


def predict_image(image_path: str, top_k: int = 5) -> dict[str, Any]:
    model = load_model_once()

    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        values, indices = torch.topk(probs, k=top_k, dim=1)

    predictions = []

    for prob, idx in zip(values[0].cpu().tolist(), indices[0].cpu().tolist()):
        class_index = int(idx)

        predictions.append(
            format_prediction_item(
                class_index=class_index,
                probability=float(prob),
            )
        )

    return {
        "top1": predictions[0],
        "top_k": predictions,
    }
