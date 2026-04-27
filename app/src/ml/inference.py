from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from ml.tiny_imagenet_resnet import ImageNetNN


MODEL_PATH = Path("models/tiny_imagenet_resnet/checkpoint.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None


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
        idx = int(idx)

        predictions.append(
            {
                "class_index": idx,
                "class_name": str(idx),
                "probability": round(float(prob), 6),
            }
        )

    return {
        "top1": predictions[0],
        "top_k": predictions,
    }
